"""
Tests for monetization.py using synthetic SceneCut data (no video file
needed). This lets us verify the filtering/scoring logic in isolation
from PySceneDetect.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scene_detector import SceneCut
from src.monetization import (
    filter_fatigue_cuts,
    filter_edge_cuts,
    score_cuts,
    enforce_minimum_ad_spacing,
    build_ad_break_windows,
)


def make_cuts(timestamps):
    return [SceneCut(timestamp_sec=t, frame_num=int(t * 30)) for t in timestamps]


def test_fatigue_filter_collapses_rapid_cuts():
    cuts = make_cuts([1.0, 1.5, 1.8, 10.0, 10.2, 30.0])
    result = filter_fatigue_cuts(cuts, min_gap_sec=3.0)
    result_times = [c.timestamp_sec for c in result]
    assert result_times == [1.0, 10.0, 30.0]


def test_edge_filter_removes_boundary_cuts():
    cuts = make_cuts([2.0, 20.0, 50.0, 98.0])
    result = filter_edge_cuts(cuts, video_duration_sec=100.0, edge_margin_sec=15.0)
    result_times = [c.timestamp_sec for c in result]
    assert result_times == [20.0, 50.0]


def test_score_cuts_favors_isolated_cuts():
    # cut at 50 is far from neighbors -> should score higher than
    # cuts bunched at 10, 12, 14
    cuts = make_cuts([10.0, 12.0, 14.0, 50.0, 90.0])
    scored = score_cuts(cuts)
    scores_by_time = {w.timestamp_sec: w.score for w in scored}
    assert scores_by_time[50.0] > scores_by_time[12.0]


def test_minimum_spacing_keeps_higher_scored_window():
    cuts = make_cuts([10.0, 15.0, 100.0])
    scored = score_cuts(cuts)
    spaced = enforce_minimum_ad_spacing(scored, min_spacing_sec=30.0)
    spaced_times = sorted(w.timestamp_sec for w in spaced)
    # 10 and 15 are within 30s of each other -> only one should survive
    assert len(spaced_times) == 2
    assert 100.0 in spaced_times


def test_full_pipeline_runs_end_to_end():
    cuts = make_cuts([1.0, 1.2, 20.0, 21.0, 60.0, 61.5, 90.0, 118.0])
    windows = build_ad_break_windows(cuts, video_duration_sec=120.0)
    assert isinstance(windows, list)
    for w in windows:
        assert 0.0 <= w.score <= 1.0
        assert 15.0 <= w.timestamp_sec <= 105.0  # respects default edge margin


if __name__ == "__main__":
    test_fatigue_filter_collapses_rapid_cuts()
    test_edge_filter_removes_boundary_cuts()
    test_score_cuts_favors_isolated_cuts()
    test_minimum_spacing_keeps_higher_scored_window()
    test_full_pipeline_runs_end_to_end()
    print("All tests passed.")
