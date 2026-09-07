"""
production_quality.py

Pre-release QC signals, all computed with established signal-processing
formulas -- no learned models, no claims about audience reaction. This
is the same category of check broadcast QC pipelines run before content
goes live: every number here has a ground-truth formula you can point
to directly.

Covers:
  - Sharpness / focus        (Laplacian variance per frame)
  - Exposure                 (histogram: blown highlights / crushed blacks)
  - Stability                (optical-flow jitter, frame-to-frame)
  - Audio quality             (clipping ratio, noise floor, silence/dropouts)
  - Color consistency         (histogram correlation across cuts -> bad splice detector)
  - Loudness compliance       (RMS-based dBFS approximation vs. a broadcast target)
  - Opening hook pacing flag  (is there a motion/cut event early enough to hook viewers)

Note on loudness: true broadcast compliance (ATSC A/85, EBU R128) uses
K-weighted integrated LUFS, which needs a proper loudness-metering
library (e.g. `pyloudnorm`) for a certified measurement. The function
here documents that distinction explicitly and is a defensible RMS-based
approximation, not a certified LUFS reading.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Visual signals (operate on individual frames -> pure, testable with
# synthetic numpy arrays, no video I/O dependency in these functions)
# ---------------------------------------------------------------------------

def compute_sharpness(frame_gray: np.ndarray) -> float:
    """
    Laplacian-variance sharpness metric. Low variance = flat, blurry
    image; high variance = lots of high-frequency edge detail.
    Standard, widely used blur-detection heuristic.
    """
    if cv2 is not None:
        lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
        return float(lap.var())
    # Pure-numpy fallback (simple discrete Laplacian kernel) so this
    # function still works without opencv installed.
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(frame_gray.astype(np.float64), 1, mode="edge")
    lap = (
        kernel[0, 1] * padded[:-2, 1:-1] + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1] + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 1] * padded[2:, 1:-1]
    )
    return float(lap.var())


@dataclass
class ExposureStats:
    mean_brightness: float          # 0-255
    pct_blown_highlights: float     # fraction of pixels >= 250
    pct_crushed_blacks: float       # fraction of pixels <= 5


def compute_exposure_stats(frame_gray: np.ndarray) -> ExposureStats:
    """Histogram-based exposure check on a grayscale frame."""
    total = frame_gray.size
    blown = float(np.count_nonzero(frame_gray >= 250)) / total
    crushed = float(np.count_nonzero(frame_gray <= 5)) / total
    return ExposureStats(
        mean_brightness=float(np.mean(frame_gray)),
        pct_blown_highlights=round(blown, 4),
        pct_crushed_blacks=round(crushed, 4),
    )


def compute_stability_jitter(flow_magnitudes: List[float]) -> float:
    """
    Given a sequence of mean optical-flow magnitudes between
    consecutive frames, return the standard deviation as a jitter
    score. Stable/tripod or smooth gimbal footage has low frame-to-
    frame variance in flow magnitude; shaky handheld footage spikes
    unevenly.
    """
    if not flow_magnitudes:
        return 0.0
    return float(np.std(flow_magnitudes))


def compute_color_consistency(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Correlation (0-1) between two color histograms (e.g. HSV hue
    histograms of the last frame of one scene and the first frame of
    the next). A sudden drop indicates a grading mismatch or a bad
    splice between two source clips.
    """
    hist_a = hist_a.astype(np.float64).flatten()
    hist_b = hist_b.astype(np.float64).flatten()
    correlation = cv2.compareHist(
        hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL
    ) if cv2 is not None else float(
        np.corrcoef(hist_a, hist_b)[0, 1]
    )
    # HISTCMP_CORREL / pearson correlation range roughly [-1, 1]; clip to [0, 1]
    return round(max(0.0, min(1.0, (correlation + 1) / 2 if correlation < 0 else correlation)), 3)


# ---------------------------------------------------------------------------
# Audio signals (operate on a numpy array of samples, mono, float [-1, 1])
# ---------------------------------------------------------------------------

def compute_audio_clipping_ratio(samples: np.ndarray, clip_threshold: float = 0.99) -> float:
    """Fraction of samples at or beyond the clipping threshold."""
    if samples.size == 0:
        return 0.0
    return round(float(np.count_nonzero(np.abs(samples) >= clip_threshold)) / samples.size, 5)


def compute_audio_noise_floor_dbfs(samples: np.ndarray, percentile: float = 5.0) -> float:
    """
    Estimate the noise floor as the dBFS level of the quietest
    `percentile` of the signal's absolute amplitude -- a proxy for
    hiss/hum/room noise underlying the actual content.
    """
    if samples.size == 0:
        return -np.inf
    abs_samples = np.abs(samples)
    quiet_threshold = np.percentile(abs_samples, percentile)
    quiet_samples = abs_samples[abs_samples <= quiet_threshold]
    rms = np.sqrt(np.mean(quiet_samples ** 2)) if quiet_samples.size else 1e-10
    rms = max(rms, 1e-10)
    return round(float(20 * np.log10(rms)), 2)


@dataclass
class SilenceWindow:
    start_sec: float
    end_sec: float


def detect_silence_dropouts(
    samples: np.ndarray,
    sample_rate: int,
    silence_threshold_dbfs: float = -50.0,
    min_duration_sec: float = 1.0,
) -> List[SilenceWindow]:
    """
    Find contiguous stretches below `silence_threshold_dbfs` lasting
    at least `min_duration_sec`. Flags both intentional silence
    (dramatic pause, fine) and unintentional audio dropouts (bad,
    needs review) -- the report should surface these for a human to
    distinguish, not auto-judge which is which.
    """
    if samples.size == 0:
        return []

    window_size = max(1, int(sample_rate * 0.05))  # 50ms analysis windows
    n_windows = samples.size // window_size
    is_silent = []
    for i in range(n_windows):
        chunk = samples[i * window_size:(i + 1) * window_size]
        rms = np.sqrt(np.mean(chunk ** 2)) if chunk.size else 0.0
        dbfs = 20 * np.log10(max(rms, 1e-10))
        is_silent.append(dbfs < silence_threshold_dbfs)

    windows = []
    start_idx = None
    for i, silent in enumerate(is_silent + [False]):  # sentinel to flush trailing window
        if silent and start_idx is None:
            start_idx = i
        elif not silent and start_idx is not None:
            duration = (i - start_idx) * window_size / sample_rate
            if duration >= min_duration_sec:
                windows.append(
                    SilenceWindow(
                        start_sec=round(start_idx * window_size / sample_rate, 2),
                        end_sec=round(i * window_size / sample_rate, 2),
                    )
                )
            start_idx = None
    return windows


def compute_loudness_dbfs_rms(samples: np.ndarray) -> float:
    """
    RMS-based loudness approximation in dBFS. NOTE: this is not a
    certified LUFS measurement (that requires K-weighting + gating per
    ITU-R BS.1770 / EBU R128) -- treat it as a fast proxy, and use a
    dedicated library like `pyloudnorm` if certified compliance
    numbers are required for delivery.
    """
    if samples.size == 0:
        return -np.inf
    rms = np.sqrt(np.mean(samples ** 2))
    rms = max(rms, 1e-10)
    return round(float(20 * np.log10(rms)), 2)


@dataclass
class LoudnessComplianceResult:
    measured_dbfs: float
    target_dbfs: float
    tolerance_db: float
    compliant: bool
    delta_db: float


def check_loudness_compliance(
    samples: np.ndarray,
    target_dbfs: float = -23.0,   # approximates EBU R128's -23 LUFS target
    tolerance_db: float = 2.0,
) -> LoudnessComplianceResult:
    """
    Compare measured RMS loudness against a broadcast target with a
    tolerance window. See compute_loudness_dbfs_rms's note on the
    RMS-vs-LUFS distinction.
    """
    measured = compute_loudness_dbfs_rms(samples)
    delta = round(measured - target_dbfs, 2)
    return LoudnessComplianceResult(
        measured_dbfs=measured,
        target_dbfs=target_dbfs,
        tolerance_db=tolerance_db,
        compliant=abs(delta) <= tolerance_db,
        delta_db=delta,
    )


# ---------------------------------------------------------------------------
# Opening hook pacing flag
# ---------------------------------------------------------------------------

@dataclass
class OpeningHookFlag:
    has_early_cut: bool
    has_early_motion_spike: bool
    first_cut_sec: Optional[float]
    hook_window_sec: float
    flagged: bool          # True = potential weak opening, worth a human look
    reason: str


def check_opening_hook_pacing(
    cuts,
    motion_series: List[Tuple[float, float]],
    hook_window_sec: float = 5.0,
    motion_spike_threshold: float = 0.5,  # in the same units as motion_series values
) -> OpeningHookFlag:
    """
    Checks whether the opening of the video (first `hook_window_sec`
    seconds) contains either a scene cut or a motion spike -- a cheap,
    grounded proxy for "does the opening have any visual event to
    grab attention," used as a QC flag rather than a performance
    prediction. A static, cut-free opening isn't necessarily bad
    (some formats open slow deliberately) -- it's flagged for a human
    to confirm that's intentional.
    """
    first_cut_sec = cuts[0].timestamp_sec if cuts else None
    has_early_cut = first_cut_sec is not None and first_cut_sec <= hook_window_sec

    has_early_motion_spike = any(
        t <= hook_window_sec and m >= motion_spike_threshold for t, m in motion_series
    )

    flagged = not (has_early_cut or has_early_motion_spike)
    reason = (
        "No cut or motion spike detected in the opening window -- confirm "
        "a slow open is intentional."
        if flagged
        else "Opening contains a cut and/or motion spike within the hook window."
    )

    return OpeningHookFlag(
        has_early_cut=has_early_cut,
        has_early_motion_spike=has_early_motion_spike,
        first_cut_sec=first_cut_sec,
        hook_window_sec=hook_window_sec,
        flagged=flagged,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class SceneProductionQuality:
    scene_start_sec: float
    sharpness: float
    exposure: ExposureStats
    stability_jitter: float
    color_consistency_to_prev: Optional[float]
    quality_score: float   # 0-100, combined
    flags: List[str] = field(default_factory=list)


def score_scene_quality(
    sharpness: float,
    exposure: ExposureStats,
    stability_jitter: float,
    color_consistency_to_prev: Optional[float],
    sharpness_ref: float = 100.0,
    jitter_ref: float = 5.0,
) -> Tuple[float, List[str]]:
    """
    Combine the raw per-scene metrics into a single 0-100 quality
    score plus human-readable flags. The reference constants
    (`sharpness_ref`, `jitter_ref`) are normalization points, not
    learned thresholds -- tune them against your own footage's
    typical range rather than treating them as universal.
    """
    flags = []

    sharpness_score = min(sharpness / sharpness_ref, 1.0) * 100
    if sharpness_score < 30:
        flags.append("Low sharpness -- possible focus/blur issue.")

    exposure_penalty = (exposure.pct_blown_highlights + exposure.pct_crushed_blacks) * 100
    exposure_score = max(0.0, 100 - exposure_penalty * 4)
    if exposure.pct_blown_highlights > 0.05:
        flags.append("Significant blown highlights.")
    if exposure.pct_crushed_blacks > 0.05:
        flags.append("Significant crushed blacks.")

    stability_score = max(0.0, 100 - (stability_jitter / jitter_ref) * 100)
    if stability_jitter > jitter_ref:
        flags.append("High frame-to-frame jitter -- possible shaky footage.")

    scores = [sharpness_score, exposure_score, stability_score]

    if color_consistency_to_prev is not None:
        color_score = color_consistency_to_prev * 100
        if color_consistency_to_prev < 0.5:
            flags.append("Large color/grading shift from previous scene -- check for a bad splice.")
        scores.append(color_score)

    overall = round(float(np.mean(scores)), 1)
    return overall, flags