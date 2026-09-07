"""
dead_air_flag.py

Dead-Air / Low-Engagement Flag.

Broadcast QC practice flags "dead air" as audio silence beyond a
tolerance window (see e.g. FCC EAS/loudness-monitoring guidance and
standard broadcast automation "silence sensor" alarms, typically
10-30s thresholds). We extend that industry baseline slightly: a
silent window that *also* has low visual motion/cut activity is a
much stronger low-engagement signal than silence alone (silence
during a fast-cutting visual montage is often intentional -- a beat
drop, a dramatic pause -- while silence during a visually static shot
usually means genuinely "nothing is happening").

This reuses:
  - production_quality.detect_silence_dropouts for the audio side
  - pacing_timeline's motion/cut-density signals for the visual side

and simply correlates the two in time. No new detection logic is
introduced here -- this module is a combinator over existing signals,
consistent with how script_reviewer.py combines other modules'
outputs into one report.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.production_quality import SilenceWindow, detect_silence_dropouts


@dataclass
class DeadAirFlag:
    start_sec: float
    end_sec: float
    duration_sec: float
    audio_silent: bool
    visual_energy_score: float   # 0-1, from the pacing timeline's energy_score
    severity: str                 # "low", "medium", "high"
    reason: str


def _energy_during(
    window: SilenceWindow,
    pacing_points,   # List[pacing_timeline.PacingPoint]
    window_sec: float,
) -> float:
    """
    Average the pacing timeline's energy_score across whichever
    pacing buckets overlap this silence window. Returns 1.0 (i.e.
    "assume high energy, don't flag") if no pacing data covers the
    window, since we shouldn't claim low engagement without visual
    evidence.
    """
    overlapping = [
        p.energy_score for p in pacing_points
        if p.timestamp_sec < window.end_sec and (p.timestamp_sec + window_sec) > window.start_sec
    ]
    return sum(overlapping) / len(overlapping) if overlapping else 1.0


def build_dead_air_flags(
    audio_samples,
    audio_sample_rate: int,
    pacing_points,
    pacing_window_sec: float = 10.0,
    silence_threshold_dbfs: float = -50.0,
    min_silence_duration_sec: float = 1.0,
    low_energy_threshold: float = 0.25,
) -> List[DeadAirFlag]:
    """
    Full entry point: find silence windows, cross-reference each
    against the pacing timeline's visual energy, and grade severity.

      - Silence + low visual energy  -> "high" (genuine dead air)
      - Silence + moderate energy    -> "medium" (likely intentional pause)
      - Silence + high visual energy -> "low" (probably a deliberate
        beat -- e.g. a dramatic hold on an active shot; still surfaced
        for a human to confirm, not auto-dismissed)
    """
    silence_windows = detect_silence_dropouts(
        audio_samples, audio_sample_rate,
        silence_threshold_dbfs=silence_threshold_dbfs,
        min_duration_sec=min_silence_duration_sec,
    )

    flags = []
    for w in silence_windows:
        energy = _energy_during(w, pacing_points, pacing_window_sec)
        duration = round(w.end_sec - w.start_sec, 2)

        if energy <= low_energy_threshold:
            severity = "high"
            reason = (
                f"{duration}s of silence with low visual energy "
                f"(score {energy:.2f}) -- likely genuine dead air, not an "
                "intentional dramatic pause."
            )
        elif energy <= 0.5:
            severity = "medium"
            reason = (
                f"{duration}s of silence with moderate visual energy "
                f"(score {energy:.2f}) -- possibly an intentional pause; "
                "worth a quick human check."
            )
        else:
            severity = "low"
            reason = (
                f"{duration}s of silence during a visually active shot "
                f"(energy {energy:.2f}) -- likely a deliberate beat, "
                "flagged for confirmation only."
            )

        flags.append(
            DeadAirFlag(
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                duration_sec=duration,
                audio_silent=True,
                visual_energy_score=round(energy, 3),
                severity=severity,
                reason=reason,
            )
        )
    return flags