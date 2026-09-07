"""
pacing_timeline.py

Content Energy / Pacing Timeline.

Combines two grounded signals into a per-window "energy" score across
the video:
  1. Cut density: how many scene cuts occur per rolling time window.
     Busy cutting = fast pacing (action, montage); sparse cutting =
     slow pacing (dialogue, exposition).
  2. Motion magnitude: average optical-flow magnitude per sampled
     frame. High on-screen motion = high energy; static/talking-head
     shots = low energy.

Both signals are classic, well-understood video-analysis proxies with
no learned/predictive component -- this is a descriptive timeline of
the content's own pacing, not a claim about how viewers will react to
it. That distinction matters: it can legitimately inform ad-placement
and QC decisions (e.g. "don't cut into an ad in the middle of a
high-energy sequence"), and it can be offered as a candidate signal
for future calibration against real retention data -- but it should
not itself be described as engagement prediction.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.scene_detector import SceneCut

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency at import time
    cv2 = None


@dataclass
class PacingPoint:
    """Energy score for one time bucket of the video."""
    timestamp_sec: float
    cut_density_score: float   # 0-1, normalized cuts-per-window
    motion_score: float        # 0-1, normalized optical-flow magnitude
    energy_score: float        # 0-1, combined score
    label: str                 # "high energy" / "medium energy" / "low energy"


def compute_cut_density(
    cuts: List[SceneCut],
    duration_sec: float,
    window_sec: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Bucket the video into `window_sec`-wide windows and count cuts per
    window. Returns (window_start_sec, raw_cut_count) pairs.
    """
    if duration_sec <= 0:
        return []

    n_windows = max(1, int(np.ceil(duration_sec / window_sec)))
    counts = [0] * n_windows

    for cut in cuts:
        idx = min(int(cut.timestamp_sec // window_sec), n_windows - 1)
        counts[idx] += 1

    return [(i * window_sec, float(c)) for i, c in enumerate(counts)]


def compute_motion_energy(
    video_path: str,
    sample_rate_hz: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Sample frames at `sample_rate_hz` and compute dense optical flow
    (Farneback) between consecutive samples. Returns (timestamp_sec,
    mean_flow_magnitude) pairs.

    Requires opencv (cv2). This does its own frame decoding since
    optical flow needs consecutive frames at a controlled interval --
    unlike object_ad_matcher, which only needs one keyframe per scene.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for motion energy computation.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / sample_rate_hz)))

    results: List[Tuple[float, float]] = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                timestamp = frame_idx / fps
                results.append((timestamp, float(np.mean(magnitude))))
            prev_gray = gray
        frame_idx += 1

    cap.release()
    return results


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values)
    if max_v <= 0:
        return [0.0 for _ in values]
    return [min(v / max_v, 1.0) for v in values]


def build_pacing_timeline(
    cuts: List[SceneCut],
    duration_sec: float,
    motion_series: List[Tuple[float, float]],
    window_sec: float = 10.0,
    cut_weight: float = 0.5,
    motion_weight: float = 0.5,
) -> List[PacingPoint]:
    """
    Combine cut density and motion magnitude into a single pacing
    timeline, bucketed by `window_sec`.

    `motion_series` is the output of compute_motion_energy (injected
    rather than computed here so this stays testable with synthetic
    data, matching the pattern used in monetization.py's tests).
    """
    density_pairs = compute_cut_density(cuts, duration_sec, window_sec)
    if not density_pairs:
        return []

    n_windows = len(density_pairs)
    density_raw = [d for _, d in density_pairs]
    density_norm = _normalize(density_raw)

    motion_raw = [0.0] * n_windows
    for t, m in motion_series:
        idx = min(int(t // window_sec), n_windows - 1)
        motion_raw[idx] = max(motion_raw[idx], m)
    motion_norm = _normalize(motion_raw)

    points = []
    for i, (window_start, _) in enumerate(density_pairs):
        energy = cut_weight * density_norm[i] + motion_weight * motion_norm[i]
        if energy > 0.66:
            label = "high energy"
        elif energy > 0.33:
            label = "medium energy"
        else:
            label = "low energy"
        points.append(
            PacingPoint(
                timestamp_sec=window_start,
                cut_density_score=round(density_norm[i], 3),
                motion_score=round(motion_norm[i], 3),
                energy_score=round(energy, 3),
                label=label,
            )
        )
    return points