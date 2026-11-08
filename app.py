import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import json
import base64
import shutil
import threading
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from core.model_manager import list_available_models, ensure_model, get_model_info
from core.video_utils import get_video_info, extract_frames_to_dir, format_timecode
from core.sam2_engine import SAM2Engine
from core.exporter import (
    export_black_and_white_matte_video,
    export_png_matte_sequence,
    export_green_screen_cutout,
    export_transparent_cutout_mov,
    export_transparent_png_sequence,
    export_single_frame_png,
    export_black_background_cutout_mp4,
    export_transparent_cutout_webm,
    export_side_by_side_alpha_mp4
)

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
FRAMES_CACHE_DIR = INPUTS_DIR / ".frames"

for d in [INPUTS_DIR, OUTPUTS_DIR, MODELS_DIR, FRAMES_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SAM 2 Unlimited Rotoscoping Studio")

# Mount static directories
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs_files", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs_files")

# Application State
class AppState:
    def __init__(self):
        self.engine = SAM2Engine(model_key="base_plus")
        self.current_video_path: Optional[str] = None
        self.current_video_name: Optional[str] = None
        self.current_frames_dir: Optional[str] = None
        self.video_info: Optional[dict] = None
        self.active_model_key = "base_plus"
        self.tracking_status = {
            "is_tracking": False,
            "current_frame": 0,
            "total_frames": 0,
            "percent": 0.0,
            "error": None,
            "done": False
        }
        self.download_status = {
            "is_downloading": False,
            "percent": 0.0,
            "model_key": None
        }

state = AppState()

# Pydantic Request Models
class PointPromptRequest(BaseModel):
    frame_idx: int
    x: float
    y: float
    label: int  # 1 for positive, 0 for negative

class ClearPromptRequest(BaseModel):
    frame_idx: int

class ExportRequest(BaseModel):
    format: str  # "transparent_png", "cutout_mp4", "single_png", "transparent_mov", "transparent_webm", "matte_video", "png_sequence", "green_screen"
    filename: Optional[str] = None
    frame_idx: Optional[int] = 0

class TrackRequest(BaseModel):
    direction: Optional[str] = "both"  # "both", "forward", "backward"
    start_frame: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/system_info")
async def get_system_info():
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Slow)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if cuda_available else 0
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "models": list_available_models(),
        "active_model": state.active_model_key
    }

@app.post("/api/models/select")
async def select_model(model_key: str = Form(...)):
    info = get_model_info(model_key)
    if not info["downloaded"]:
        raise HTTPException(status_code=400, detail="Model is not downloaded yet. Please download it first.")
    
    ckpt_path, config_file = ensure_model(model_key)
    state.engine = SAM2Engine(model_key=model_key)
    state.engine.load_model(ckpt_path, config_file)
    state.active_model_key = model_key

    # Re-init video state if a video is currently loaded
    if state.current_frames_dir and state.video_info:
        state.engine.init_video(state.current_frames_dir, state.video_info["total_frames"])

    return {"status": "ok", "active_model": model_key}

@app.post("/api/models/download")
async def trigger_model_download(model_key: str = Form(...), background_tasks: BackgroundTasks = None):
    def download_task():
        state.download_status["is_downloading"] = True
        state.download_status["model_key"] = model_key
        def prog(down, total):
            if total > 0:
                state.download_status["percent"] = round((down / total) * 100, 1)
        try:
            ckpt_path, cfg = ensure_model(model_key, progress_callback=prog)
            # Auto load
            state.engine.load_model(ckpt_path, cfg)
            state.active_model_key = model_key
        except Exception as e:
            print(f"Error downloading model: {e}")
        finally:
            state.download_status["is_downloading"] = False

    background_tasks.add_task(download_task)
    return {"status": "download_started", "model_key": model_key}

@app.get("/api/models/download_progress")
async def get_download_progress():
    return state.download_status

@app.get("/api/videos/list")
async def list_videos():
    valid_exts = {".mp4", ".mov", ".avi", ".mkv"}
    videos = []
    for f in INPUTS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts:
            size_mb = round(f.stat().st_size / (1024 * 1024), 1)
            videos.append({"filename": f.name, "path": str(f), "size_mb": size_mb})
    return {"videos": videos}

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        dest_path = INPUTS_DIR / file.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer, length=8 * 1024 * 1024)
        return {"filename": file.filename, "path": str(dest_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@app.post("/api/open_inputs_folder")
async def open_inputs_folder():
    try:
        os.startfile(str(INPUTS_DIR))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/open_outputs_folder")
async def open_outputs_folder():
    try:
        os.startfile(str(OUTPUTS_DIR))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/videos/load")
async def load_video(filename: str = Form(...)):
    video_path = INPUTS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    state.current_video_path = str(video_path)
    state.current_video_name = video_path.stem
    state.current_frames_dir = str(FRAMES_CACHE_DIR / state.current_video_name)

    # 1. Get metadata
    info = get_video_info(state.current_video_path)
    state.video_info = info

    # 2. Extract frames to cache
    extract_frames_to_dir(state.current_video_path, state.current_frames_dir)

    # 3. Ensure model loaded
    try:
        ckpt_path, config_file = ensure_model(state.active_model_key)
        if state.engine.predictor is None:
            state.engine.load_model(ckpt_path, config_file)
    except Exception as e:
        print(f"Model load note: {e}")

    # 4. Init video state in SAM 2 with CPU offloading
    if state.engine.predictor is not None:
        state.engine.init_video(state.current_frames_dir, info["total_frames"])

    return {"status": "loaded", "info": info}

@app.get("/api/frame/{frame_idx}")
async def get_frame(frame_idx: int):
    if not state.current_frames_dir:
        raise HTTPException(status_code=400, detail="No video loaded.")

    frame_file = Path(state.current_frames_dir) / f"{frame_idx:05d}.jpg"
    if not frame_file.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")

    return FileResponse(str(frame_file), media_type="image/jpeg")

def mask_to_png_base64(mask_np: np.ndarray, color=(0, 229, 255), alpha=160) -> str:
    """
    Converts binary mask (H, W) uint8 (0 or 255) to a transparent RGBA PNG base64 string.
    Selected pixels (255) get the cyan tint + alpha.
    Unselected pixels (0) are completely transparent (alpha = 0).
    """
    h, w = mask_np.shape[:2]
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    selected = (mask_np > 0)
    bgra[selected, 0] = color[2]  # B: 255
    bgra[selected, 1] = color[1]  # G: 229
    bgra[selected, 2] = color[0]  # R: 0
    bgra[selected, 3] = alpha     # Alpha (0-255)
    _, buffer = cv2.imencode(".png", bgra)
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/api/mask/{frame_idx}")
async def get_mask_overlay(frame_idx: int):
    """Returns the generated mask as an RGBA PNG image with transparency for the web canvas."""
    mask = state.engine.get_mask(frame_idx)
    if mask is None:
        return JSONResponse({"has_mask": False})

    b64_str = mask_to_png_base64(mask)
    return {"has_mask": True, "mask_b64": b64_str}

@app.post("/api/prompt/add")
async def add_prompt(req: PointPromptRequest):
    if not state.current_frames_dir or state.video_info is None:
        raise HTTPException(status_code=400, detail="Video not loaded.")

    mask = state.engine.add_point_prompt(
        frame_idx=req.frame_idx,
        x=req.x,
        y=req.y,
        label=req.label
    )

    b64_str = mask_to_png_base64(mask)
    return {
        "status": "ok",
        "frame_idx": req.frame_idx,
        "mask_b64": b64_str,
        "prompts": state.engine.prompts.get(req.frame_idx, {})
    }

@app.post("/api/prompt/clear")
async def clear_prompt(req: ClearPromptRequest):
    state.engine.clear_frame_prompts(req.frame_idx)
    return {"status": "cleared", "frame_idx": req.frame_idx}

@app.post("/api/track/start")
async def start_tracking(background_tasks: BackgroundTasks, req: Optional[TrackRequest] = None):
    if state.engine.inference_state is None:
        raise HTTPException(status_code=400, detail="Video not loaded or model not ready.")
    if len(state.engine.prompts) == 0:
        raise HTTPException(status_code=400, detail="Please add at least one point prompt before tracking.")

    direction = req.direction if req and req.direction else "both"
    start_frame = req.start_frame if req else None

    def run_tracking():
        state.tracking_status = {
            "is_tracking": True,
            "direction": direction,
            "current_frame": 0,
            "total_frames": state.video_info["total_frames"],
            "percent": 0.0,
            "error": None,
            "done": False
        }

        def prog(cur, total, frame_idx):
            state.tracking_status["current_frame"] = frame_idx
            state.tracking_status["percent"] = min(100.0, round((cur / total) * 100, 1))

        try:
            success = state.engine.track_frames(
                direction=direction,
                start_frame_idx=start_frame,
                progress_callback=prog
            )
            state.tracking_status["done"] = success
        except Exception as e:
            state.tracking_status["error"] = str(e)
            print(f"Tracking error: {e}")
        finally:
            state.tracking_status["is_tracking"] = False

    background_tasks.add_task(run_tracking)
    return {"status": "tracking_started", "direction": direction}

@app.get("/api/track/status")
async def get_tracking_status():
    return state.tracking_status

@app.post("/api/export")
async def export_matte(req: ExportRequest):
    if len(state.engine.masks_cache) == 0:
        raise HTTPException(status_code=400, detail="No masks generated. Run tracking first.")

    v_info = state.video_info
    base_name = req.filename or state.current_video_name or "rotoscope"
    fps = v_info["fps"]
    width = v_info["width"]
    height = v_info["height"]
    total = v_info["total_frames"]

    if req.format == "transparent_png":
        out_dirname = f"{base_name}_transparent_png"
        out_dir = OUTPUTS_DIR / out_dirname
        folder, zip_path = export_transparent_png_sequence(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_dir),
            width, height, total,
            create_zip=True
        )
        zip_name = Path(zip_path).name if zip_path else None
        zip_url = f"/outputs_files/{zip_name}" if zip_name else None
        return {
            "status": "ok",
            "format": req.format,
            "file": zip_name or out_dirname,
            "url": zip_url,
            "folder": str(out_dir)
        }

    elif req.format == "cutout_mp4":
        out_filename = f"{base_name}_cutout_black.mp4"
        out_path = OUTPUTS_DIR / out_filename
        export_black_background_cutout_mp4(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format == "side_by_side_mp4":
        out_filename = f"{base_name}_alpha_packed.mp4"
        out_path = OUTPUTS_DIR / out_filename
        export_side_by_side_alpha_mp4(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format == "single_png":
        frame_idx = req.frame_idx if req.frame_idx is not None else 0
        out_filename = f"{base_name}_frame_{frame_idx:05d}_cutout.png"
        out_path = OUTPUTS_DIR / out_filename
        export_single_frame_png(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            frame_idx,
            width, height
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format in ("transparent_mp4", "transparent_mov"):
        ext = "mp4" if req.format == "transparent_mp4" else "mov"
        out_filename = f"{base_name}_cutout.{ext}"
        out_path = OUTPUTS_DIR / out_filename
        export_transparent_cutout_mov(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format == "transparent_webm":
        out_filename = f"{base_name}_cutout.webm"
        out_path = OUTPUTS_DIR / out_filename
        export_transparent_cutout_webm(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format == "matte_video":
        out_filename = f"{base_name}_matte.mp4"
        out_path = OUTPUTS_DIR / out_filename
        export_black_and_white_matte_video(
            state.engine.masks_cache,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    elif req.format == "png_sequence":
        out_dirname = f"{base_name}_png_sequence"
        out_dir = OUTPUTS_DIR / out_dirname
        folder, zip_path = export_png_matte_sequence(
            state.engine.masks_cache,
            str(out_dir),
            width, height, total,
            create_zip=True
        )
        zip_name = Path(zip_path).name if zip_path else None
        zip_url = f"/outputs_files/{zip_name}" if zip_name else None
        return {
            "status": "ok",
            "format": req.format,
            "file": zip_name or out_dirname,
            "url": zip_url,
            "folder": str(out_dir)
        }

    elif req.format == "green_screen":
        out_filename = f"{base_name}_greenscreen.mp4"
        out_path = OUTPUTS_DIR / out_filename
        export_green_screen_cutout(
            state.engine.masks_cache,
            state.current_frames_dir,
            str(out_path),
            fps, width, height, total
        )
        return {"status": "ok", "format": req.format, "file": out_filename, "url": f"/outputs_files/{out_filename}"}

    raise HTTPException(status_code=400, detail="Unknown export format.")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  SAM 2 UNLIMITED ROTOSCOPING STUDIO")
    print("  Designed for DaVinci Resolve")
    print("  Starting on http://127.0.0.1:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

