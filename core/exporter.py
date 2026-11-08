import os
import cv2
import zipfile
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

def export_black_and_white_matte_video(
    masks: Dict[int, np.ndarray],
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
):
    """
    Exports a clean Black & White Matte MP4 video file.
    In DaVinci Resolve Color Page, add as External Matte.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height), isColor=True)

    blank_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(total_frames):
        if i in masks and masks[i] is not None:
            m = masks[i]
            if len(m.shape) == 2:
                frame_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = m
        else:
            frame_bgr = blank_mask

        writer.write(frame_bgr)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    writer.release()
    print(f"Exported Matte Video: {output_path}")
    return output_path

def export_png_matte_sequence(
    masks: Dict[int, np.ndarray],
    output_dir: str,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None,
    create_zip: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Exports a sequential Black & White PNG image sequence (matte_00000.png)
    and packages into a ZIP archive for easy download.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blank_mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(total_frames):
        filename = out_dir / f"matte_{i:05d}.png"
        mask = masks.get(i, blank_mask)
        cv2.imwrite(str(filename), mask)

        if progress_callback and i % 50 == 0:
            progress_callback(i, total_frames)

    zip_file = None
    if create_zip:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.glob("*.png")):
                zf.write(p, arcname=p.name)
        zip_file = str(zip_path)

    print(f"Exported B&W PNG sequence ({total_frames} frames) to: {output_dir}")
    return str(out_dir), zip_file

def export_transparent_png_sequence(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_dir: str,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None,
    create_zip: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Exports a sequential 32-bit RGBA PNG image sequence (cutout_00000.png)
    and packages them into a ZIP file for instant browser download.
    Each PNG contains ONLY the subject with 100% transparency on the background.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = Path(frames_dir)

    for i in range(total_frames):
        filename = out_dir / f"cutout_{i:05d}.png"
        src_frame_path = frames_path / f"{i:05d}.jpg"

        if src_frame_path.exists():
            src_frame = cv2.imread(str(src_frame_path))  # BGR
            mask = masks.get(i, None)

            if mask is not None:
                alpha_channel = mask  # 0 or 255
            else:
                alpha_channel = np.zeros((height, width), dtype=np.uint8)

            # BGRA for cv2.imwrite (supports alpha channel in PNG)
            bgra = np.dstack((src_frame, alpha_channel))
            cv2.imwrite(str(filename), bgra)
        else:
            blank = np.zeros((height, width, 4), dtype=np.uint8)
            cv2.imwrite(str(filename), blank)

        if progress_callback and i % 50 == 0:
            progress_callback(i, total_frames)

    zip_file = None
    if create_zip:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.glob("*.png")):
                zf.write(p, arcname=p.name)
        zip_file = str(zip_path)

    print(f"Exported Transparent PNG sequence ({total_frames} frames) to: {output_dir}")
    return str(out_dir), zip_file

def export_single_frame_png(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    frame_idx: int,
    width: int,
    height: int
) -> str:
    """
    Exports a single frame as a 32-bit transparent RGBA PNG cutout.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    src_frame_path = Path(frames_dir) / f"{frame_idx:05d}.jpg"

    if src_frame_path.exists():
        src_frame = cv2.imread(str(src_frame_path))
        mask = masks.get(frame_idx, None)
        if mask is not None:
            alpha = mask
        else:
            alpha = np.zeros((height, width), dtype=np.uint8)
        bgra = np.dstack((src_frame, alpha))
        cv2.imwrite(str(out_file), bgra)
    else:
        blank = np.zeros((height, width, 4), dtype=np.uint8)
        cv2.imwrite(str(out_file), blank)

    print(f"Exported Single Frame PNG: {output_path}")
    return str(out_file)

def export_black_background_cutout_mp4(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
) -> str:
    """
    Exports a standard MP4 video with the subject isolated on a pure black background.
    Compatible with any media player, YouTube, and video editors (composite with Screen/Add mode).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    frames_path = Path(frames_dir)
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(total_frames):
        src_frame_path = frames_path / f"{i:05d}.jpg"
        if not src_frame_path.exists():
            writer.write(blank)
            continue

        src_frame = cv2.imread(str(src_frame_path))
        mask = masks.get(i, None)

        if mask is not None:
            if len(mask.shape) == 2:
                alpha = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
            else:
                alpha = (mask[:, :, 0] > 0).astype(np.uint8)[:, :, np.newaxis]
            cutout = src_frame * alpha
            writer.write(cutout)
        else:
            writer.write(blank)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    writer.release()
    print(f"Exported Black Background Cutout MP4: {output_path}")
    return str(out_file)

def export_side_by_side_alpha_mp4(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
) -> str:
    """
    Exports a Side-by-Side MP4 (Color Cutout on Left + Alpha Matte on Right).
    This is the industry-standard way to store a full transparent alpha channel in an MP4 container.
    In DaVinci Resolve:
    Drop the clip onto Track 2, open the Color page or Fusion, and pipe the right half as the alpha matte.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width * 2, height))
    frames_path = Path(frames_dir)
    blank_half = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(total_frames):
        src_frame_path = frames_path / f"{i:05d}.jpg"
        if not src_frame_path.exists():
            combined = np.hstack((blank_half, blank_half))
            writer.write(combined)
            continue

        src_frame = cv2.imread(str(src_frame_path))
        mask = masks.get(i, None)

        if mask is not None:
            if len(mask.shape) == 2:
                m_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                alpha_norm = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
            else:
                m_bgr = mask
                alpha_norm = (mask[:, :, 0] > 0).astype(np.uint8)[:, :, np.newaxis]

            left_rgb = src_frame * alpha_norm
            right_matte = m_bgr
            combined = np.hstack((left_rgb, right_matte))
        else:
            combined = np.hstack((blank_half, blank_half))

        writer.write(combined)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    writer.release()
    print(f"Exported Side-by-Side Alpha MP4 for DaVinci Resolve: {output_path}")
    return str(out_file)

def export_transparent_cutout_webm(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
) -> str:
    """
    Exports WebM (VP9) video with true Alpha transparency channel.
    Natively transparent in Chrome, Edge, DaVinci Resolve, Premiere, and web apps.
    """
    import av
    from fractions import Fraction

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frames_path = Path(frames_dir)

    container = av.open(str(out_file), mode='w', format='mov')
    rate = Fraction(str(round(fps, 3))) if fps > 0 else Fraction(30, 1)
    stream = container.add_stream('libvpx-vp9', rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuva420p'
    stream.options = {'auto-alt-ref': '0'}

    blank_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    for i in range(total_frames):
        src_frame_path = frames_path / f"{i:05d}.jpg"
        if not src_frame_path.exists():
            rgba = blank_rgba
        else:
            src_frame = cv2.imread(str(src_frame_path))
            mask = masks.get(i, None)
            rgb = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
            alpha = mask if mask is not None else np.zeros((height, width), dtype=np.uint8)
            rgba = np.dstack((rgb, alpha))

        frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
        frame_yuva = frame.reformat(format='yuva420p')
        for packet in stream.encode(frame_yuva):
            container.mux(packet)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Exported VP9 Transparent WebM: {output_path}")
    return str(out_file)

def export_transparent_cutout_mov(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
):
    """
    Exports Apple ProRes 4444 (.mov) with embedded Alpha channel.
    This is the industry-standard transparent video format for DaVinci Resolve.
    Drop it on Track 2 directly above your video — the background is 100% transparent,
    and only the subject is visible with zero setup required!
    """
    import av
    from fractions import Fraction

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frames_path = Path(frames_dir)

    container = av.open(str(out_file), mode='w', format='mov')
    rate = Fraction(str(round(fps, 3))) if fps > 0 else Fraction(30, 1)
    stream = container.add_stream('prores_ks', rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuva444p10le'
    stream.options = {'profile': '4'}  # Profile 4 = Apple ProRes 4444 (with alpha)

    blank_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    for i in range(total_frames):
        src_frame_path = frames_path / f"{i:05d}.jpg"
        if not src_frame_path.exists():
            rgba = blank_rgba
        else:
            src_frame = cv2.imread(str(src_frame_path))  # BGR
            mask = masks.get(i, None)

            # Convert BGR -> RGB and attach mask as alpha
            rgb = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
            if mask is not None:
                alpha = mask  # 0 or 255
            else:
                alpha = np.zeros((height, width), dtype=np.uint8)

            rgba = np.dstack((rgb, alpha))

        frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
        frame_yuva = frame.reformat(format='yuva444p10le')
        for packet in stream.encode(frame_yuva):
            container.mux(packet)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Exported ProRes 4444 Transparent MOV: {output_path}")
    return output_path

def export_green_screen_cutout(
    masks: Dict[int, np.ndarray],
    frames_dir: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    progress_callback=None
):
    """
    Exports video with subject preserved and background replaced with bright green (#00FF00)
    for instant chroma-keying in any NLE.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    frames_path = Path(frames_dir)

    for i in range(total_frames):
        src_frame_path = frames_path / f"{i:05d}.jpg"
        if not src_frame_path.exists():
            continue

        src_frame = cv2.imread(str(src_frame_path))
        mask = masks.get(i, None)

        if mask is not None:
            # Create green background
            green_bg = np.zeros_like(src_frame)
            green_bg[:, :] = (0, 255, 0)  # BGR green

            # Alpha blend mask (0 or 1)
            alpha = (mask / 255.0).astype(np.float32)[:, :, np.newaxis]
            composited = (src_frame.astype(np.float32) * alpha + green_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        else:
            composited = np.zeros_like(src_frame)
            composited[:, :] = (0, 255, 0)

        writer.write(composited)

        if progress_callback and i % 30 == 0:
            progress_callback(i, total_frames)

    writer.release()
    print(f"Exported Green Screen Video: {output_path}")
    return output_path
