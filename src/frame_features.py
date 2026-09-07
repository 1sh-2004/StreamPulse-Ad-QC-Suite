"""
frame_features.py

Low-level, per-frame computer-vision signals shared by several higher-level
modules (quality.py, pacing.py, mood.py, object_detection.py).

Everything in this file is a classic, well-understood CV metric (Laplacian
sharpness, histogram-based exposure, frame-differencing motion, optical-flow
jitter). None of it is a learned/trained model and none of it predicts
audience behavior -- it describes measurable properties of the pixels
themselves. Keeping these primitives in one place means every downstream
"score" module is built on the same, auditable ground truth.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class SampledFrame:
    """A single frame pulled from the video at a known timestamp."""
    timestamp_sec: float
    frame: np.ndarray  # BGR, as returned by OpenCV


def sample_frames(video_path: str, every_n_seconds: float = 1.0) -> List[SampledFrame]:
    """
    Uniformly sample frames from a video at a fixed time interval.

    This is the shared sampling strategy for all frame-based scoring below.
    Sampling on a time grid (rather than every frame) keeps the pipeline fast
    on long videos while still catching enough of the video to be
    representative.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0.0

    samples: List[SampledFrame] = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append(SampledFrame(timestamp_sec=t, frame=frame))
        t += every_n_seconds

    cap.release()
    return samples


def laplacian_sharpness(frame: np.ndarray) -> float:
    """
    Focus/sharpness via variance of the Laplacian. This is the standard
    classic-CV blur-detection metric: a sharp frame has lots of high-frequency
    edge content (high variance); a blurry frame is smooth (low variance).
    Raw value is unbounded (typically 0-2000+ for real footage); callers
    normalize as needed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_metrics(frame: np.ndarray) -> dict:
    """
    Histogram-based exposure check.
    Returns fractions of pixels that are "crushed" (near-black, <5) or
    "blown" (near-white, >250) in the luminance channel, plus the mean
    brightness. High crushed/blown fractions indicate under/over-exposure.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = gray.size
    crushed = float(np.sum(gray < 5)) / total
    blown = float(np.sum(gray > 250)) / total
    mean_brightness = float(np.mean(gray))
    return {
        "crushed_fraction": crushed,
        "blown_fraction": blown,
        "mean_brightness": mean_brightness,
    }


def frame_diff_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """
    Cheap motion-intensity proxy: mean absolute pixel difference between two
    consecutive sampled frames (grayscale, resized for speed). Higher value =
    more visual change between the two samples -- used as a pacing/"energy"
    signal, not a semantic understanding of what moved.
    """
    a = cv2.cvtColor(cv2.resize(prev_frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(cv2.resize(curr_frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float(np.mean(np.abs(a - b)))


def optical_flow_jitter(frames: List[np.ndarray]) -> float:
    """
    Camera-stability proxy using dense optical flow (Farneback) between
    consecutive frames. We measure the *variance* of the mean flow-vector
    magnitude across the sequence: smooth, intentional camera movement
    produces a fairly steady flow magnitude, while shaky handheld footage
    produces large frame-to-frame swings. This is a proxy for jitter, not a
    stabilization-grade motion model.

    Returns a single float; higher = shakier/less stable.
    """
    if len(frames) < 2:
        return 0.0

    grays = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY) for f in frames]
    magnitudes = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None, 0.5, 2, 15, 2, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))

    return float(np.std(magnitudes)) if magnitudes else 0.0


def dominant_color_stats(frame: np.ndarray) -> Tuple[float, float]:
    """
    Rough brightness/saturation summary of a frame in HSV space. Used as a
    weak visual proxy for "mood" (e.g. darker/desaturated vs. bright/vivid),
    not a semantic understanding of the scene's content.
    Returns (mean_saturation_0_255, mean_value_0_255).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])), float(np.mean(hsv[:, :, 2]))