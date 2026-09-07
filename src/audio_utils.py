"""
audio_utils.py

ffmpeg-based audio extraction. Responsible for exactly one thing:
given a video file, return its audio track as a mono numpy array of
float samples in [-1, 1], plus the sample rate -- the exact shape
`production_quality.py`'s audio functions, `brand_safety.py`'s
`flag_audio_spikes`, and `dead_air_flag.py`'s `build_dead_air_flags`
all expect.

Extraction happens once in app.py and the resulting (samples, rate)
pair is injected into every audio-dependent module, matching the
"inject the expensive I/O" pattern used elsewhere in this pipeline
(pacing_timeline's motion_series, script_reviewer's frame_sampler).

Requires ffmpeg + ffprobe on PATH (not pip-installable -- see
requirements.txt). No Python audio-decoding dependency (audioread,
pydub, etc.) is needed: we shell out to ffmpeg and read its raw PCM
output directly.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Sample rate we ask ffmpeg to resample to. 22.05kHz is plenty for the
# RMS/dBFS-based signals this project uses (production_quality,
# brand_safety, dead_air_flag) -- none of them need full 44.1/48kHz
# fidelity, and a lower rate keeps the extracted array small for long
# videos.
DEFAULT_SAMPLE_RATE = 22050


class AudioExtractionError(Exception):
    """
    Raised when a video's audio track can't be extracted -- missing
    ffmpeg/ffprobe binaries, no audio stream in the file, or ffmpeg
    itself failing on a malformed/unsupported file. Callers (see
    app.py) are expected to catch this and degrade gracefully: every
    audio-dependent module in this pipeline already treats
    audio_samples=None as "skip the audio signal," not a hard error.
    """
    pass


@dataclass
class AudioTrackInfo:
    has_audio: bool
    codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_sec: Optional[float] = None


def _require_ffmpeg_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise AudioExtractionError(
            f"Required binary(ies) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (e.g. `apt-get install ffmpeg` / `brew install "
            "ffmpeg`) -- see requirements.txt."
        )


def probe_audio_track(video_path: str) -> AudioTrackInfo:
    """
    Cheap metadata check via ffprobe: does this file have an audio
    stream at all, and if so what does it look like. Useful for
    surfacing a clear "no audio track" message in the UI before
    attempting a full extraction (or decoding to PCM unnecessarily).

    Does not raise on "no audio stream" -- that's a normal, expected
    case (e.g. silent B-roll), reflected as `has_audio=False`. Only
    raises AudioExtractionError if ffprobe itself can't run at all.
    """
    _require_ffmpeg_binaries()

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,duration",
        "-of", "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as e:  # pragma: no cover -- covered by _require_ffmpeg_binaries
        raise AudioExtractionError(f"ffprobe not found: {e}")
    except subprocess.TimeoutExpired:
        raise AudioExtractionError(f"ffprobe timed out probing '{video_path}'.")

    if proc.returncode != 0:
        raise AudioExtractionError(
            f"ffprobe failed on '{video_path}': {proc.stderr.strip()}"
        )

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise AudioExtractionError(f"Could not parse ffprobe output: {e}")

    streams = data.get("streams") or []
    if not streams:
        return AudioTrackInfo(has_audio=False)

    stream = streams[0]
    duration = stream.get("duration")
    sample_rate = stream.get("sample_rate")
    return AudioTrackInfo(
        has_audio=True,
        codec=stream.get("codec_name"),
        sample_rate=int(sample_rate) if sample_rate is not None else None,
        channels=stream.get("channels"),
        duration_sec=float(duration) if duration is not None else None,
    )


def extract_audio(
    video_path: str,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Extract the video's audio track as mono float32 PCM samples in
    [-1, 1], resampled to `target_sample_rate`.

    This is the shared entry point every audio-dependent module in
    this pipeline is written against:
      - production_quality.py's clipping/noise-floor/silence/loudness
        functions
      - brand_safety.py's flag_audio_spikes
      - dead_air_flag.py's build_dead_air_flags (via
        production_quality.detect_silence_dropouts)

    Implementation: shells out to ffmpeg to decode + resample + downmix
    directly to raw 16-bit PCM on stdout (`-f s16le`), then converts to
    float32 in [-1, 1] with numpy. This avoids writing an intermediate
    .wav file to disk and avoids any Python-side audio-decoding
    dependency beyond numpy.

    Returns:
        (samples, sample_rate) where samples is a 1-D mono float32
        numpy array in [-1, 1].

    Raises:
        AudioExtractionError if ffmpeg/ffprobe aren't available, the
        file has no audio stream, or ffmpeg fails to decode it.
    """
    _require_ffmpeg_binaries()

    info = probe_audio_track(video_path)
    if not info.has_audio:
        raise AudioExtractionError(
            f"'{video_path}' has no audio stream to extract."
        )

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vn",                          # no video
        "-ac", "1",                     # downmix to mono
        "-ar", str(target_sample_rate), # resample
        "-f", "s16le",                  # raw signed 16-bit little-endian PCM
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except FileNotFoundError as e:  # pragma: no cover -- covered by _require_ffmpeg_binaries
        raise AudioExtractionError(f"ffmpeg not found: {e}")
    except subprocess.TimeoutExpired:
        raise AudioExtractionError(f"ffmpeg timed out extracting audio from '{video_path}'.")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioExtractionError(f"ffmpeg failed on '{video_path}': {stderr}")

    raw = proc.stdout
    if not raw:
        raise AudioExtractionError(
            f"ffmpeg produced no audio data for '{video_path}' "
            "(empty/corrupt audio stream?)."
        )

    # int16 PCM -> float32 in [-1, 1]. 32768.0 (not 32767.0) is the
    # standard int16-to-float normalization divisor; it keeps the
    # conversion symmetric and matches what every other float-PCM
    # tool (numpy, scipy.io.wavfile, ffmpeg's own f32le output) uses.
    int_samples = np.frombuffer(raw, dtype="<i2")
    float_samples = (int_samples.astype(np.float32) / 32768.0)

    return float_samples, target_sample_rate