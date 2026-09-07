"""
brand_safety.py

Brand Safety Flag (YOLOv8 + audio).

Two independent, grounded signals, combined into a per-scene flag a
human reviewer can act on. Deliberately conservative: this is a
"needs review" signal, not an automated block/allow decision -- brand
safety calls carry real commercial risk and should stay human-in-the-loop.

  1. Visual: reuse the same YOLOv8 detector as object_ad_matcher.py to
     flag COCO object classes that commonly co-occur with brand-unsafe
     content in advertising contexts (weapons-adjacent objects,
     alcohol containers, etc.). COCO's vocabulary is narrow and has no
     "weapon" or "violence" class, so this catches only what's in that
     80-class vocabulary -- it is a coarse pre-filter, not a
     comprehensive brand-safety classifier.
  2. Audio: reuse production_quality.py's audio primitives (clipping
     ratio, RMS loudness) to flag scenes with an abnormal loudness
     spike relative to the video's own baseline -- a cheap proxy for
     "something sonically aggressive happened here" (shouting,
     gunfire-like impacts, alarms), worth a human's ears.

Both signals key off scene timestamps so results line up with the
other per-scene modules (production_quality, pacing_timeline).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.object_ad_matcher import DetectedObject, detect_objects_in_frame, load_detector
from src.scene_detector import SceneCut

# COCO classes with a defensible "flag for brand-safety review" mapping.
# Conservative and narrow by design -- see module docstring.
RISKY_COCO_CLASSES: Dict[str, str] = {
    "knife": "Sharp object in frame",
    "scissors": "Sharp object in frame",
    "wine glass": "Alcohol context",
    "bottle": "Possible alcohol context (bottle -- not alcohol-specific in COCO)",
}


@dataclass
class VisualSafetyFlag:
    scene_start_sec: float
    triggered_classes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class AudioSafetyFlag:
    window_start_sec: float
    window_end_sec: float
    peak_dbfs: float
    baseline_dbfs: float
    delta_db: float
    reason: str


@dataclass
class SceneBrandSafety:
    scene_start_sec: float
    visual_flag: Optional[VisualSafetyFlag]
    audio_flags: List[AudioSafetyFlag]
    flagged: bool          # True = worth a human review pass
    reason: str


def flag_visual_risk(
    detections: List[DetectedObject],
    scene_start_sec: float,
) -> Optional[VisualSafetyFlag]:
    """
    Check one scene's frame detections against RISKY_COCO_CLASSES.
    Returns None if nothing in this scene matches -- callers should
    treat "no flag" as "nothing in our narrow vocabulary triggered,"
    not as a clean bill of health.
    """
    triggered, reasons = [], []
    for det in detections:
        if det.label in RISKY_COCO_CLASSES:
            triggered.append(det.label)
            reasons.append(RISKY_COCO_CLASSES[det.label])
    if not triggered:
        return None
    return VisualSafetyFlag(
        scene_start_sec=scene_start_sec,
        triggered_classes=sorted(set(triggered)),
        reasons=sorted(set(reasons)),
    )


def flag_audio_spikes(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float = 1.0,
    spike_threshold_db: float = 12.0,
) -> List[AudioSafetyFlag]:
    """
    Bucket audio into `window_sec` windows, compute RMS dBFS per
    window, and flag windows that spike more than `spike_threshold_db`
    above the video's own median loudness. A relative (video-specific)
    threshold is used instead of an absolute one because "loud" is
    content-dependent -- an action trailer's baseline is louder than a
    dialogue scene's.
    """
    if samples.size == 0:
        return []

    window_size = max(1, int(sample_rate * window_sec))
    n_windows = samples.size // window_size
    if n_windows == 0:
        return []

    window_dbfs = []
    for i in range(n_windows):
        chunk = samples[i * window_size:(i + 1) * window_size]
        rms = np.sqrt(np.mean(chunk ** 2)) if chunk.size else 0.0
        window_dbfs.append(20 * np.log10(max(rms, 1e-10)))

    baseline = float(np.median(window_dbfs))
    flags = []
    for i, dbfs in enumerate(window_dbfs):
        delta = dbfs - baseline
        if delta >= spike_threshold_db:
            start = i * window_sec
            flags.append(
                AudioSafetyFlag(
                    window_start_sec=round(start, 2),
                    window_end_sec=round(start + window_sec, 2),
                    peak_dbfs=round(dbfs, 2),
                    baseline_dbfs=round(baseline, 2),
                    delta_db=round(delta, 2),
                    reason=(
                        f"Loudness spike {delta:.1f}dB above the video's baseline -- "
                        "worth a human check (shout, impact, alarm, etc.)."
                    ),
                )
            )
    return flags


def build_brand_safety_report(
    video_path: str,
    scene_cuts: List[SceneCut],
    frame_sampler,
    audio_samples: Optional[np.ndarray] = None,
    audio_sample_rate: Optional[int] = None,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.35,
) -> List[SceneBrandSafety]:
    """
    Full pipeline entry point. Runs the visual detector per scene
    keyframe (reusing object_ad_matcher's detector/loader so we don't
    load YOLO twice in the same process) and, if audio was supplied,
    maps audio-spike windows onto the scene they fall inside.

    `audio_samples`/`audio_sample_rate` are injected (from
    audio_utils.extract_audio) rather than extracted here, matching
    the "inject the expensive I/O" pattern used elsewhere in this
    pipeline (pacing_timeline's motion_series, script_reviewer's
    frame_sampler).
    """
    detector = load_detector(model_name)

    audio_flags_all: List[AudioSafetyFlag] = []
    if audio_samples is not None and audio_sample_rate:
        audio_flags_all = flag_audio_spikes(audio_samples, audio_sample_rate)

    results = []
    sorted_cuts = sorted(scene_cuts, key=lambda c: c.timestamp_sec)
    for i, cut in enumerate(sorted_cuts):
        scene_start = cut.timestamp_sec
        scene_end = (
            sorted_cuts[i + 1].timestamp_sec if i + 1 < len(sorted_cuts) else float("inf")
        )

        frame = frame_sampler(video_path, scene_start)
        visual_flag = None
        if frame is not None:
            detections = detect_objects_in_frame(detector, frame, confidence_threshold)
            visual_flag = flag_visual_risk(detections, scene_start)

        scene_audio_flags = [
            f for f in audio_flags_all if scene_start <= f.window_start_sec < scene_end
        ]

        flagged = visual_flag is not None or bool(scene_audio_flags)
        reason_parts = []
        if visual_flag:
            reason_parts.append(f"Visual: {', '.join(visual_flag.reasons)}")
        if scene_audio_flags:
            reason_parts.append(f"Audio: {len(scene_audio_flags)} loudness spike(s)")
        reason = "; ".join(reason_parts) if reason_parts else "No brand-safety signals triggered."

        results.append(
            SceneBrandSafety(
                scene_start_sec=scene_start,
                visual_flag=visual_flag,
                audio_flags=scene_audio_flags,
                flagged=flagged,
                reason=reason,
            )
        )
    return results