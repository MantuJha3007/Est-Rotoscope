import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SAM2Engine:
    def __init__(self, model_key: str = "base_plus"):
        self.model_key = model_key
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_state = None
        self.video_frames_dir = None
        self.total_frames = 0
        self.masks_cache: Dict[int, np.ndarray] = {}  # frame_idx -> binary mask (H, W) uint8
        self.prompts: Dict[int, Dict] = {}  # frame_idx -> {"points": [(x, y)], "labels": [1 or 0]}
        self.is_tracking = False

    def load_model(self, checkpoint_path: str, config_file: str):
        """Loads SAM 2 Video Predictor with optimal precision and device settings."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM 2 is not installed in this environment. "
                "Run `pip install -e .` from the SAM 2 directory or install `sam-2`."
            )

        print(f"Loading SAM 2 predictor ({self.model_key}) on device: {self.device}...")
        
        # Configure torch backends for NVIDIA RTX
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device
        )
        print("SAM 2 Predictor loaded successfully.")
        return True

    def init_video(self, frames_dir: str, total_frames: int):
        """
        Initializes video inference state for SAM 2.
        CRITICAL: offload_video_to_cpu=True and offload_state_to_cpu=True ensure
        arbitrary clip lengths (>30s, 5 min, etc.) run seamlessly without GPU OOM.
        """
        if self.predictor is None:
            raise RuntimeError("Model is not loaded. Call load_model first.")

        self.video_frames_dir = frames_dir
        self.total_frames = total_frames
        self.masks_cache.clear()
        self.prompts.clear()

        # Free previous state if any
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Video frame & state offloading:
        # Offload video frames and memory state to CPU memory to prevent CUDA Out-of-Memory (OOM)
        # on 8GB consumer GPUs, especially for long or 1080p clips.
        # All neural network forward passes, attention, and mask predictions STILL RUN ON GPU.
        print(f"Initializing inference state for {total_frames} frames with memory safety on {self.device}...")

        self.inference_state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True
        )
        print("Inference state ready.")
        return True

    def add_point_prompt(self, frame_idx: int, x: float, y: float, label: int, obj_id: int = 1):
        """
        Adds a positive (label=1) or negative (label=0) click prompt to a specific frame.
        Coordinates x, y are in pixels (image coordinate space).
        """
        if self.inference_state is None:
            raise RuntimeError("Video state not initialized.")

        if frame_idx not in self.prompts:
            self.prompts[frame_idx] = {"points": [], "labels": []}

        self.prompts[frame_idx]["points"].append([x, y])
        self.prompts[frame_idx]["labels"].append(label)

        points = np.array(self.prompts[frame_idx]["points"], dtype=np.float32)
        labels = np.array(self.prompts[frame_idx]["labels"], dtype=np.int32)

        # Autocast on CUDA for efficiency
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.no_grad()
        )

        with autocast_context:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # Convert logits to binary mask (H, W) uint8 (0 or 255)
        # out_mask_logits shape: [1, 1, H, W]
        mask_np = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
        self.masks_cache[frame_idx] = mask_np
        return mask_np

    def clear_frame_prompts(self, frame_idx: int, obj_id: int = 1):
        """Clears prompts and mask for a specific frame."""
        if frame_idx in self.prompts:
            del self.prompts[frame_idx]
        if frame_idx in self.masks_cache:
            del self.masks_cache[frame_idx]

        if self.inference_state is not None:
            # Re-init inference state or reset obj
            try:
                self.predictor.reset_state(self.inference_state)
            except Exception:
                pass
            # Re-apply any other keyframe prompts
            for f_idx, p_data in self.prompts.items():
                if f_idx != frame_idx and len(p_data["points"]) > 0:
                    pts = np.array(p_data["points"], dtype=np.float32)
                    lbls = np.array(p_data["labels"], dtype=np.int32)
                    self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=f_idx,
                        obj_id=obj_id,
                        points=pts,
                        labels=lbls,
                    )

    def track_frames(
        self,
        direction: str = "both",
        start_frame_idx: Optional[int] = None,
        progress_callback=None,
        cancel_check=None
    ):
        """
        Propagates segmentation across the video in specified direction:
        - "both": propagates forward to end AND backward to frame 0 (Full Bidirectional Tracking)
        - "forward": propagates from start_frame_idx forward to the end of the video
        - "backward": propagates from start_frame_idx backward to frame 0
        """
        if self.inference_state is None:
            raise RuntimeError("Inference state not initialized.")

        self.is_tracking = True
        total = self.total_frames

        # Determine starting frame if not provided
        if start_frame_idx is None:
            if self.prompts:
                start_frame_idx = min(self.prompts.keys())
            else:
                start_frame_idx = 0

        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.no_grad()
        )

        steps_done = 0
        if direction == "both":
            total_steps = total
        elif direction == "forward":
            total_steps = max(1, total - start_frame_idx)
        else: # backward
            total_steps = max(1, start_frame_idx + 1)

        def run_propagation(rev: bool, s_idx: int):
            nonlocal steps_done
            with autocast_context:
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=s_idx,
                    reverse=rev
                ):
                    if cancel_check and cancel_check():
                        print("Tracking cancelled by user.")
                        return False

                    mask_np = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    self.masks_cache[out_frame_idx] = mask_np

                    steps_done += 1
                    if out_frame_idx > 0 and out_frame_idx % 40 == 0:
                        if self.device == "cuda":
                            torch.cuda.empty_cache()

                    if progress_callback:
                        progress_callback(steps_done, total_steps, out_frame_idx)
            return True

        print(f"Tracking ({direction}) from frame {start_frame_idx} (total {total} frames)...")

        # Execute tracking based on direction
        if direction in ("forward", "both"):
            ok = run_propagation(rev=False, s_idx=start_frame_idx)
            if not ok:
                self.is_tracking = False
                return False

        if direction in ("backward", "both"):
            if start_frame_idx > 0 or direction == "backward":
                ok = run_propagation(rev=True, s_idx=start_frame_idx)
                if not ok:
                    self.is_tracking = False
                    return False

        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.is_tracking = False
        print(f"Tracking finished ({direction}). Produced {len(self.masks_cache)} masks.")
        return True

    def track_all_frames(self, progress_callback=None, cancel_check=None):
        """Backward-compatible alias for bidirectional tracking."""
        return self.track_frames(direction="both", progress_callback=progress_callback, cancel_check=cancel_check)

    def get_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Returns the binary mask (0 or 255) for a given frame."""
        return self.masks_cache.get(frame_idx, None)

