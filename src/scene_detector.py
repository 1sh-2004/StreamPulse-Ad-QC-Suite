"""
scene_detector.py

Thin wrapper around PySceneDetect. Responsible for exactly one thing:
given a video file, return the list of timestamps (in seconds) where
a scene change / content break occurs.

This intentionally does NOT contain any monetization logic — that
lives in monetization.py. Keeping detection and business logic
separate makes both easier to reason about and test independently.
"""

from dataclasses import dataclass
from typing import List

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


@dataclass
class SceneCut:
    """A single detected scene boundary."""
    timestamp_sec: float
    frame_num: int


def detect_scenes(video_path: str, threshold: float = 27.0) -> List[SceneCut]:
    """
    Run PySceneDetect's content-aware detector on a video file and
    return the list of scene-cut points.

    Args:
        video_path: path to an MP4 (or other supported) video file.
        threshold: sensitivity for ContentDetector. Lower = more cuts
            detected (more sensitive to smaller visual changes).
            27.0 is PySceneDetect's own recommended default for
            typical film/TV content.

    Returns:
        List of SceneCut objects, sorted by timestamp ascending.
        Note: PySceneDetect returns *scene boundaries*, i.e. the start
        of each new scene. We treat each boundary as a candidate cut
        point for ad insertion.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    cuts = []
    for scene_start, _scene_end in scene_list:
        cuts.append(
            SceneCut(
                timestamp_sec=scene_start.get_seconds(),
                frame_num=scene_start.get_frames(),
            )
        )

    return cuts


def get_video_duration(video_path: str) -> float:
    """Return total duration of the video in seconds."""
    video = open_video(video_path)
    # duration is only known after we've scanned, so fall back to
    # video.duration if the backend exposes it directly
    return video.duration.get_seconds() if video.duration else 0.0