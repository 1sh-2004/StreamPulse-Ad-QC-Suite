"""
object_ad_matcher.py

Contextual object detection -> ad category matching.

Uses YOLOv8 (via the `ultralytics` package) to detect objects in
sampled frames, then maps the detected COCO object classes to
advertising categories using a static lookup table.

Design notes:
  - This is deliberately NOT a prediction of "what ad will perform
    best." It is a grounded contextual-relevance signal: "this scene
    visually contains a kitchen/food-related object, so a food &
    beverage ad is contextually relevant here" -- the same principle
    behind contextual ad networks (e.g. Google's content-based
    contextual targeting), just applied to video frames instead of
    page text.
  - COCO's 80 classes are a limited vocabulary. The mapping below only
    covers classes with a reasonably confident ad-category mapping;
    everything else is left unmapped rather than force-fit.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.scene_detector import SceneCut

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency at import time
    YOLO = None


# COCO class name -> advertising category. Deliberately conservative:
# only classes with a clear, defensible mapping are included.
COCO_TO_AD_CATEGORY: Dict[str, str] = {
    "cup": "Food & Beverage", "bottle": "Food & Beverage", "wine glass": "Food & Beverage",
    "banana": "Food & Beverage", "apple": "Food & Beverage", "sandwich": "Food & Beverage",
    "orange": "Food & Beverage", "pizza": "Food & Beverage", "cake": "Food & Beverage",
    "donut": "Food & Beverage", "bowl": "Food & Beverage", "fork": "Food & Beverage",
    "knife": "Food & Beverage", "spoon": "Food & Beverage",

    "laptop": "Technology & Electronics", "cell phone": "Technology & Electronics",
    "tv": "Technology & Electronics", "keyboard": "Technology & Electronics",
    "mouse": "Technology & Electronics", "remote": "Technology & Electronics",

    "car": "Automotive", "truck": "Automotive", "motorcycle": "Automotive", "bus": "Automotive",

    "dog": "Pet Products", "cat": "Pet Products",

    "sports ball": "Sports & Fitness", "tennis racket": "Sports & Fitness",
    "baseball bat": "Sports & Fitness", "skateboard": "Sports & Fitness",
    "surfboard": "Sports & Fitness", "skis": "Sports & Fitness",
    "snowboard": "Sports & Fitness",

    "handbag": "Fashion & Retail", "backpack": "Fashion & Retail",
    "tie": "Fashion & Retail", "suitcase": "Fashion & Retail",

    "couch": "Home & Furniture", "bed": "Home & Furniture",
    "dining table": "Home & Furniture", "potted plant": "Home & Furniture",

    "book": "Media & Education",
}


@dataclass
class DetectedObject:
    """A single object detection in a sampled frame."""
    label: str
    confidence: float
    bbox_xyxy: List[float]


@dataclass
class AdCategoryMatch:
    """Aggregated ad-category relevance for one scene."""
    category: str
    confidence: float          # 0-1, share of supporting detections weighted by their confidence
    supporting_objects: List[str] = field(default_factory=list)


@dataclass
class SceneAdMatch:
    """Contextual ad-category matches for a single scene window."""
    scene_start_sec: float
    matches: List[AdCategoryMatch]


def load_detector(model_name: str = "yolov8n.pt"):
    """
    Load a YOLOv8 model. Uses the nano checkpoint by default (fast
    enough for per-scene keyframe sampling; accuracy can be traded up
    to yolov8s/m if latency isn't a constraint).
    """
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Run `pip install ultralytics` "
            "to enable object-based contextual ad matching."
        )
    return YOLO(model_name)


def detect_objects_in_frame(
    detector, frame: np.ndarray, confidence_threshold: float = 0.35
) -> List[DetectedObject]:
    """
    Run YOLOv8 inference on a single frame (BGR or RGB numpy array,
    HxWx3) and return detections above `confidence_threshold`.
    """
    results = detector(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        cls_id = int(box.cls[0])
        label = results.names.get(cls_id, str(cls_id))
        xyxy = box.xyxy[0].tolist()
        detections.append(DetectedObject(label=label, confidence=conf, bbox_xyxy=xyxy))
    return detections


def match_objects_to_ad_categories(
    detections: List[DetectedObject],
) -> List[AdCategoryMatch]:
    """
    Aggregate a list of frame-level detections into ranked ad-category
    matches. Confidence per category = sum of matching detection
    confidences, normalized so the top category isn't artificially
    capped at a single detection's score.
    """
    if not detections:
        return []

    category_weight: Dict[str, float] = Counter()
    category_objects: Dict[str, List[str]] = {}

    for det in detections:
        category = COCO_TO_AD_CATEGORY.get(det.label)
        if category is None:
            continue
        category_weight[category] += det.confidence
        category_objects.setdefault(category, []).append(det.label)

    if not category_weight:
        return []

    max_weight = max(category_weight.values())
    matches = [
        AdCategoryMatch(
            category=cat,
            confidence=round(min(weight / max_weight, 1.0), 3),
            supporting_objects=sorted(set(category_objects[cat])),
        )
        for cat, weight in category_weight.items()
    ]
    return sorted(matches, key=lambda m: m.confidence, reverse=True)


def build_contextual_ad_matches(
    video_path: str,
    scene_cuts: List[SceneCut],
    frame_sampler,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.35,
) -> List[SceneAdMatch]:
    """
    Full pipeline entry point: for each scene, sample its keyframe,
    run object detection, and map detections to ad categories.

    Args:
        video_path: path to the source video.
        scene_cuts: scene boundaries from scene_detector.detect_scenes.
        frame_sampler: callable(video_path, timestamp_sec) -> np.ndarray,
            returning the frame at that timestamp. Injected rather than
            hard-coded so this module has no direct video-decoding
            dependency of its own (keeps it testable in isolation).
        model_name: YOLOv8 checkpoint to use.
        confidence_threshold: minimum detection confidence to keep.
    """
    detector = load_detector(model_name)
    results = []
    for cut in scene_cuts:
        frame = frame_sampler(video_path, cut.timestamp_sec)
        if frame is None:
            continue
        detections = detect_objects_in_frame(detector, frame, confidence_threshold)
        matches = match_objects_to_ad_categories(detections)
        results.append(SceneAdMatch(scene_start_sec=cut.timestamp_sec, matches=matches))
    return results