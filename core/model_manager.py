import os
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "tiny": {
        "filename": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "description": "Fastest (~150MB), ideal for fast preview and high FPS"
    },
    "small": {
        "filename": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "description": "Fast & lightweight (~185MB)"
    },
    "base_plus": {
        "filename": "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "description": "Recommended default (~320MB), excellent accuracy and speed"
    },
    "large": {
        "filename": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "description": "Highest quality (~900MB), max precision for fine details"
    }
}

def get_model_info(model_key="base_plus"):
    if model_key not in MODEL_CONFIGS:
        model_key = "base_plus"
    info = MODEL_CONFIGS[model_key].copy()
    local_path = MODELS_DIR / info["filename"]
    info["path"] = str(local_path)
    info["downloaded"] = local_path.exists() and local_path.stat().st_size > 10_000_000
    return info

def list_available_models():
    res = {}
    for k in MODEL_CONFIGS:
        res[k] = get_model_info(k)
    return res

def ensure_model(model_key="base_plus", progress_callback=None):
    """
    Ensures model weights exist in models/ directory.
    Checks existing local downloads or downloads from official Meta release.
    """
    info = get_model_info(model_key)
    target_path = Path(info["path"])

    if info["downloaded"]:
        return str(target_path), info["config"]

    # Check if user already has it in kvn_rotoscope weights
    kvn_weights = Path(r"C:\Users\mantu\AppData\Local\kvn_rotoscope\weights") / info["filename"]
    if kvn_weights.exists() and kvn_weights.stat().st_size > 10_000_000:
        try:
            print(f"Reusing existing weights from {kvn_weights}...")
            shutil.copy2(kvn_weights, target_path)
            return str(target_path), info["config"]
        except Exception as e:
            print(f"Could not copy from {kvn_weights}: {e}")

    # Download from URL
    url = info["url"]
    print(f"Downloading {info['filename']} from Meta ({url})...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB
    downloaded = 0

    with open(target_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=info["filename"]) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    print(f"Download complete: {target_path}")
    return str(target_path), info["config"]

