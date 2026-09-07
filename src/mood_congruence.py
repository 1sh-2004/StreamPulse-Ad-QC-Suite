"""
mood_congruence.py

Mood Congruence Score.

Advertising research on the "mood-congruence effect" (e.g. Goldberg &
Gorn, 1987, "Happy and Sad TV Programs: How They Affect Reactions to
Commercials", Journal of Consumer Research; and the broader affect-
priming literature it spawned) has repeatedly found that ad recall and
brand attitude are measurably affected by whether an ad's emotional
tone matches the surrounding content's tone -- an upbeat ad embedded
in a somber scene tends to underperform relative to the same ad placed
in tonally matched content. This module gives a grounded (non-learned)
proxy for that match.

Two pieces:
  1. A coarse per-scene "visual mood" label derived from
     frame_features.dominant_color_stats (mean saturation/brightness
     in HSV). This is explicitly a WEAK proxy -- brightness/saturation
     correlate with perceived mood in film-lighting convention (dark +
     desaturated reads as somber/tense; bright + saturated reads as
     upbeat/energetic) but this is a heuristic, not a validated
     affect-recognition model. It is combined with the pacing
     timeline's energy_score, which the mood-congruence literature
     also links to arousal, for a slightly less naive estimate.
  2. A static ad-category -> expected-mood table (reusing
     object_ad_matcher's ad categories), so a scene's mood can be
     compared against the mood a given ad category is conventionally
     associated with, producing a 0-1 congruence score.

As with pacing_timeline.py, this stays a descriptive signal ("here is
this scene's estimated mood, and how well it matches this ad
category's conventional mood") -- not a claim to predict how a viewer
will actually feel.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.frame_features import dominant_color_stats

# Coarse mood labels, ordered low -> high on both axes for scoring.
MOOD_LABELS = ["somber", "calm", "neutral", "upbeat", "energetic"]

# Ad category -> conventional expected mood (index into MOOD_LABELS).
# Deliberately coarse and editable -- these are defensible defaults,
# not a claim of universal truth; a real deployment should let brand
# teams override per-category expectations.
AD_CATEGORY_EXPECTED_MOOD: Dict[str, str] = {
    "Food & Beverage": "upbeat",
    "Technology & Electronics": "neutral",
    "Automotive": "energetic",
    "Pet Products": "upbeat",
    "Sports & Fitness": "energetic",
    "Fashion & Retail": "upbeat",
    "Home & Furniture": "calm",
    "Media & Education": "neutral",
}


@dataclass
class ScenedMood:
    scene_start_sec: float
    saturation: float          # 0-255, raw mean HSV saturation
    brightness: float          # 0-255, raw mean HSV value
    energy_score: Optional[float]   # 0-1, from pacing_timeline, if available
    mood_label: str


@dataclass
class MoodCongruenceResult:
    scene_start_sec: float
    scene_mood: str
    ad_category: str
    expected_mood: str
    congruence_score: float    # 0-1, 1 = same label, decays with label distance
    reason: str


def classify_scene_mood(
    saturation: float,
    brightness: float,
    energy_score: Optional[float] = None,
) -> str:
    """
    Map raw HSV stats (+ optional pacing energy) onto one of
    MOOD_LABELS. Saturation/brightness are normalized against typical
    8-bit frame ranges; energy_score (already 0-1) nudges the estimate
    toward "energetic" when visual pacing is high, since arousal is
    part of what "energetic" is meant to capture here.
    """
    # Normalize brightness/saturation to roughly 0-1.
    b_norm = min(brightness / 255.0, 1.0)
    s_norm = min(saturation / 255.0, 1.0)
    base = (b_norm + s_norm) / 2.0

    if energy_score is not None:
        base = 0.7 * base + 0.3 * energy_score

    idx = min(int(base * len(MOOD_LABELS)), len(MOOD_LABELS) - 1)
    return MOOD_LABELS[idx]


def score_mood_congruence(
    scene_start_sec: float,
    scene_mood: str,
    ad_category: str,
) -> MoodCongruenceResult:
    """
    Score how well a scene's estimated mood matches an ad category's
    expected mood, using distance along MOOD_LABELS. Same label = 1.0;
    each step away decays the score. Unknown ad categories fall back
    to "neutral" expectation rather than raising, since new categories
    will show up as object_ad_matcher's COCO mapping is extended.
    """
    expected_mood = AD_CATEGORY_EXPECTED_MOOD.get(ad_category, "neutral")

    try:
        scene_idx = MOOD_LABELS.index(scene_mood)
        expected_idx = MOOD_LABELS.index(expected_mood)
    except ValueError:
        scene_idx = expected_idx = MOOD_LABELS.index("neutral")

    distance = abs(scene_idx - expected_idx)
    max_distance = len(MOOD_LABELS) - 1
    score = round(1.0 - (distance / max_distance), 3)

    if distance == 0:
        reason = f"Scene mood ({scene_mood}) matches {ad_category}'s expected mood exactly."
    elif distance == 1:
        reason = (
            f"Scene mood ({scene_mood}) is adjacent to {ad_category}'s expected "
            f"mood ({expected_mood}) -- minor mismatch."
        )
    else:
        reason = (
            f"Scene mood ({scene_mood}) is far from {ad_category}'s expected "
            f"mood ({expected_mood}) -- likely a tonal mismatch, per the "
            "mood-congruence literature this ad may underperform here."
        )

    return MoodCongruenceResult(
        scene_start_sec=scene_start_sec,
        scene_mood=scene_mood,
        ad_category=ad_category,
        expected_mood=expected_mood,
        congruence_score=score,
        reason=reason,
    )


def build_mood_congruence_report(
    video_path: str,
    scene_cuts: List,
    frame_sampler,
    scene_ad_matches: List,           # List[object_ad_matcher.SceneAdMatch]
    pacing_points: Optional[List] = None,   # List[pacing_timeline.PacingPoint]
    pacing_window_sec: float = 10.0,
) -> List[MoodCongruenceResult]:
    """
    Full pipeline entry point: for each scene that has at least one
    matched ad category (from object_ad_matcher.build_contextual_ad_matches),
    estimate the scene's mood and score congruence against its
    top-ranked ad category.
    """
    energy_by_window: Dict[float, float] = {}
    if pacing_points:
        for p in pacing_points:
            energy_by_window[p.timestamp_sec] = p.energy_score

    match_by_scene = {m.scene_start_sec: m for m in scene_ad_matches}

    results = []
    for cut in scene_cuts:
        scene_start = cut.timestamp_sec
        match = match_by_scene.get(scene_start)
        if not match or not match.matches:
            continue

        frame = frame_sampler(video_path, scene_start)
        if frame is None:
            continue

        saturation, brightness = dominant_color_stats(frame)

        energy_score = None
        if energy_by_window:
            window_key = max(
                (t for t in energy_by_window if t <= scene_start),
                default=None,
            )
            if window_key is not None:
                energy_score = energy_by_window[window_key]

        mood_label = classify_scene_mood(saturation, brightness, energy_score)
        top_category = match.matches[0].category
        results.append(score_mood_congruence(scene_start, mood_label, top_category))

    return results