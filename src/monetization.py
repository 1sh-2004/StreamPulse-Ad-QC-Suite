"""
monetization.py

Converts raw scene-cut timestamps (from scene_detector.py) into a
ranked list of ad-break windows.

Design goals / rules encoded here:
  1. Ad-fatigue avoidance: reject any cut that is too close to the
     previous accepted cut (default: 3 seconds). Rapid cuts usually
     mean fast-paced action, not a natural narrative break -- inserting
     an ad there feels jarring.
  2. Edge avoidance: don't place ad breaks too close to the very start
     or end of the video (viewers dropped mid-intro/outro churn hard).
  3. Minimum spacing between ad breaks: even after filtering cuts, we
     don't want ad windows bunched together -- enforce a minimum gap
     between *accepted ad windows* (default: configurable, e.g. 60s),
     which approximates "don't show two ad breaks 10 seconds apart."
  4. Scoring: not all remaining cuts are equally good. We score each
     candidate using how "isolated" it is -- i.e. how far it sits from
     its neighboring cuts. A cut that sits in a long gap between other
     cuts is more likely to be a genuine scene/act break than one that's
     just one of many cuts in a busy sequence.
"""

from dataclasses import dataclass
from typing import List
from src.scene_detector import SceneCut


@dataclass
class AdBreakWindow:
    """A single recommended ad-insertion point."""
    timestamp_sec: float
    score: float          # 0-1, higher = better ad-break candidate
    reason: str           # human-readable justification


def filter_fatigue_cuts(
    cuts: List[SceneCut],
    min_gap_sec: float = 3.0,
) -> List[SceneCut]:
    """
    Rule 1: Ad-fatigue filter.
    Drop any cut that occurs less than `min_gap_sec` after the
    previously *accepted* cut. This collapses rapid-fire cut
    sequences (action scenes, quick edits) down to a single
    representative point, since inserting ads mid-action is exactly
    the kind of experience JioStar wants to avoid.
    """
    if not cuts:
        return []

    accepted = [cuts[0]]
    for cut in cuts[1:]:
        if cut.timestamp_sec - accepted[-1].timestamp_sec >= min_gap_sec:
            accepted.append(cut)

    return accepted


def filter_edge_cuts(
    cuts: List[SceneCut],
    video_duration_sec: float,
    edge_margin_sec: float = 15.0,
) -> List[SceneCut]:
    """
    Rule 2: Edge avoidance.
    Remove cuts that fall within `edge_margin_sec` of the start or
    end of the video. Viewers are most sensitive to interruptions
    right after starting a video or right before it ends.
    """
    return [
        c for c in cuts
        if edge_margin_sec <= c.timestamp_sec <= (video_duration_sec - edge_margin_sec)
    ]


def score_cuts(cuts: List[SceneCut]) -> List[AdBreakWindow]:
    """
    Rule 4: Score each remaining cut by how "isolated" it is from its
    neighbors. A cut sitting in a long stretch with no other cuts
    nearby is more likely to represent an act break / natural pause
    than a scene transition. We use the average distance to the
    previous and next cut as a proxy for this, normalized to 0-1
    across the candidate set.

    This is a simple heuristic, not a learned model -- but it gives a
    principled way to rank candidates rather than treating every
    surviving cut as equally good.
    """
    if not cuts:
        return []

    n = len(cuts)
    raw_scores = []

    for i, cut in enumerate(cuts):
        prev_gap = cut.timestamp_sec - cuts[i - 1].timestamp_sec if i > 0 else None
        next_gap = cuts[i + 1].timestamp_sec - cut.timestamp_sec if i < n - 1 else None

        gaps = [g for g in (prev_gap, next_gap) if g is not None]
        # If this is the only cut left (no neighbors to compare against),
        # treat it as maximally isolated -- there's nothing nearby to
        # suggest it's part of a busy cut sequence.
        isolation = sum(gaps) / len(gaps) if gaps else 1.0
        raw_scores.append(isolation)

    max_score = max(raw_scores) if raw_scores else 1.0
    max_score = max_score if max_score > 0 else 1.0

    windows = []
    for cut, raw in zip(cuts, raw_scores):
        normalized = min(raw / max_score, 1.0)
        reason = (
            "Isolated scene break -- long gap from neighboring cuts, "
            "likely a natural act/scene boundary."
            if normalized > 0.5
            else "Scene break with nearby cuts -- acceptable but lower confidence."
        )
        windows.append(
            AdBreakWindow(timestamp_sec=cut.timestamp_sec, score=round(normalized, 3), reason=reason)
        )

    return windows


def enforce_minimum_ad_spacing(
    windows: List[AdBreakWindow],
    min_spacing_sec: float = 60.0,
) -> List[AdBreakWindow]:
    """
    Rule 3: Minimum spacing between accepted ad breaks.
    Even after fatigue + edge filtering, greedily keep the
    highest-scored window within each `min_spacing_sec` block rather
    than allowing multiple ad breaks close together.
    """
    if not windows:
        return []

    # Sort by timestamp for the spacing pass, but track score to break ties.
    sorted_windows = sorted(windows, key=lambda w: w.timestamp_sec)

    accepted: List[AdBreakWindow] = [sorted_windows[0]]
    for w in sorted_windows[1:]:
        if w.timestamp_sec - accepted[-1].timestamp_sec >= min_spacing_sec:
            accepted.append(w)
        elif w.score > accepted[-1].score:
            # Replace the last accepted window if this nearby one scores higher.
            accepted[-1] = w

    return accepted


def build_ad_break_windows(
    cuts: List[SceneCut],
    video_duration_sec: float,
    fatigue_gap_sec: float = 3.0,
    edge_margin_sec: float = 15.0,
    min_ad_spacing_sec: float = 60.0,
) -> List[AdBreakWindow]:
    """
    Full monetization pipeline: raw cuts -> filtered, scored,
    spaced-out ad-break recommendations.

    This is the single entry point app.py should call.
    """
    filtered = filter_fatigue_cuts(cuts, min_gap_sec=fatigue_gap_sec)
    filtered = filter_edge_cuts(filtered, video_duration_sec, edge_margin_sec=edge_margin_sec)
    scored = score_cuts(filtered)
    final = enforce_minimum_ad_spacing(scored, min_spacing_sec=min_ad_spacing_sec)

    # Best candidates first for display purposes.
    return sorted(final, key=lambda w: w.score, reverse=True)