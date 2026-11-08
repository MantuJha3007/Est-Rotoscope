import os
import cv2
import json
from pathlib import Path

def get_video_info(video_path: str):
    """Returns video metadata: fps, frame_count, width, height, duration in seconds."""
    # First try with PyAV (supports ProRes/APV, HEVC, H.264, DNxHD, etc.)
    try:
        import av
        with av.open(video_path) as container:
            if container.streams.video:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                if fps <= 0 or fps > 240:
                    fps = 30.0
                width = stream.width
                height = stream.height
                total_frames = int(stream.frames) if stream.frames else 0
                duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0.0
                if duration <= 0 and container.duration:
                    duration = float(container.duration / av.time.AV_TIME_BASE)
                if total_frames <= 0 and duration > 0 and fps > 0:
                    total_frames = int(round(duration * fps))

                return {
                    "fps": round(fps, 3),
                    "total_frames": total_frames,
                    "width": width,
                    "height": height,
                    "duration": round(duration, 2),
                    "filename": Path(video_path).name
                }
    except Exception as e:
        print(f"PyAV info failed, falling back to cv2: {e}")

    # Fallback to OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

    return {
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": round(duration, 2),
        "filename": Path(video_path).name
    }

def format_timecode(frame_idx: int, fps: float) -> str:
    """Format frame index into SMPTE timecode HH:MM:SS:FF"""
    if fps <= 0:
        fps = 30.0
    total_seconds = int(frame_idx // fps)
    frames = int(frame_idx % fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def extract_frames_to_dir(video_path: str, output_dir: str, progress_callback=None, max_fps=None):
    """
    Extracts video frames to a directory as %05d.jpg.
    SAM 2 native video predictor operates directly on a directory of JPEG images.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    total_frames = info["total_frames"]

    # Check if frames are already extracted
    existing = list(out_path.glob("*.jpg"))
    if len(existing) >= total_frames and total_frames > 0:
        print(f"Frames already extracted in {output_dir} ({len(existing)} frames)")
        info["extracted_frames"] = len(existing)
        return info

    frame_idx = 0

    # Try PyAV first (universal codec support: ProRes, HEVC, H.264, etc.)
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frame_file = out_path / f"{frame_idx:05d}.jpg"
                img = frame.to_ndarray(format="bgr24")
                cv2.imwrite(str(frame_file), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1

                if progress_callback and frame_idx % 15 == 0:
                    progress_callback(frame_idx, total_frames)

        if frame_idx > 0:
            info["extracted_frames"] = frame_idx
            info["total_frames"] = frame_idx
            print(f"Extracted {frame_idx} frames using PyAV to {output_dir}")
            return info
    except Exception as e:
        print(f"PyAV extraction encountered error: {e}, attempting cv2 fallback...")

    # Fallback to OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = out_path / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1

        if progress_callback and frame_idx % 15 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    info["extracted_frames"] = frame_idx
    if info["total_frames"] <= 0:
        info["total_frames"] = frame_idx
    print(f"Extracted {frame_idx} frames to {output_dir}")
    return info

def get_single_frame(video_path: str, frame_idx: int):
    """Reads a single frame from video without extracting the entire video."""
    try:
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i == frame_idx:
                    return frame.to_ndarray(format="bgr24")
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame
