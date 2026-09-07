# 🎬 Context-Aware Ad Insertion Engine

A tool that analyzes video content end-to-end to recommend **natural
ad-insertion points** — scene cuts and fades — instead of interrupting
mid-action or mid-dialogue, and to surface the QC, monetization, and
brand-safety signals a human reviewer needs to sign off on a video
before it goes into an ad-supported stream.

Built with VOD/streaming monetization in mind: the core goal is
maximizing ad placements while minimizing viewer disruption
("ad-fatigue"), backed up by a set of grounded, explainable signals —
not black-box scores — so every recommendation traces back to a
specific, inspectable formula or detection.

## How it works

The pipeline runs in this order (see `src/app.py`'s module docstring
for the same list with line references):

1. **Scene detection** ([`src/scene_detector.py`](src/scene_detector.py)) —
   uses [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
   (content-aware detector) to find every scene boundary in the video.
2. **Monetization filtering** ([`src/monetization.py`](src/monetization.py)) —
   the core ad-placement logic. Raw scene cuts are filtered and scored:
   - **Ad-fatigue filter**: collapses rapid-fire cuts (e.g. action
     sequences) so we don't recommend an ad break every 1–2 seconds.
   - **Edge avoidance**: no ad breaks within N seconds of video
     start/end.
   - **Isolation scoring**: cuts that sit in a long gap between other
     cuts score higher — more likely a genuine act/scene break than an
     incidental cut in a busy sequence.
   - **Minimum ad spacing**: enforces a minimum gap between recommended
     ad breaks, keeping the highest-confidence one in each window.
3. **Audio extraction** ([`src/audio_utils.py`](src/audio_utils.py)) —
   ffmpeg-based: pulls the video's audio track once as mono
   float32 PCM samples, shared by every audio-dependent module below
   (`production_quality`, `brand_safety`, `dead_air_flag`), so ffmpeg
   only runs once per video regardless of how many modules need audio.
4. **Content Energy / Pacing Timeline** ([`src/pacing_timeline.py`](src/pacing_timeline.py)) —
   combines cut density and optical-flow motion magnitude into a
   per-window "energy" score across the video (fast-cut action vs.
   slow dialogue), used both descriptively and as an input to
   `dead_air_flag.py`.
5. **Technical/Production Quality Score** ([`src/production_quality.py`](src/production_quality.py)) —
   pre-release QC signals computed with established, citable
   signal-processing formulas (no learned models): sharpness (Laplacian
   variance), exposure (blown highlights / crushed blacks), stability
   (optical-flow jitter), color-consistency across cuts (bad-splice
   detection), audio clipping/noise-floor/silence, RMS-based loudness
   compliance, and an opening-hook pacing check.
6. **Contextual object detection → ad matching** ([`src/object_ad_matcher.py`](src/object_ad_matcher.py)) —
   YOLOv8 (`ultralytics`) detects objects in sampled frames and maps
   COCO classes to advertising categories via a static, conservative
   lookup table (contextual relevance, not performance prediction).
7. **Mood Congruence Score** ([`src/mood_congruence.py`](src/mood_congruence.py)) —
   a grounded, non-learned proxy for the mood-congruence effect from
   advertising research (Goldberg & Gorn, 1987): estimates a scene's
   visual mood from color/brightness + pacing energy, and scores how
   well that matches an ad category's conventional mood.
8. **Brand Safety Flag** ([`src/brand_safety.py`](src/brand_safety.py)) —
   two independent, grounded signals combined into a per-scene "needs
   human review" flag (deliberately not an automated block/allow
   decision): YOLOv8 detections against a narrow, defensible set of
   risky COCO classes, and audio loudness spikes relative to the
   video's own baseline.
9. **Dead-Air / Low-Engagement Flag** ([`src/dead_air_flag.py`](src/dead_air_flag.py)) —
   correlates `production_quality`'s silence detection with
   `pacing_timeline`'s visual energy score in the same window, so
   silence during a static shot is flagged as likely dead air while
   silence during a visually busy shot is flagged only for
   confirmation (probably an intentional beat).
10. **Script Generation + AI Reviewer** ([`src/script_reviewer.py`](src/script_reviewer.py)) —
    optional, only runs with an Anthropic API key configured: samples a
    keyframe per scene cut, generates a scene-by-scene script via a
    vision-capable LLM, then has the LLM produce a structured
    qualitative critique (pacing, tone, narrative clarity, opening
    hook, brand-safety notes) against a fixed rubric — an editor's-note
    report, not a numeric "will perform well" score.
11. **Visualization** ([`src/app.py`](src/app.py)) — a Streamlit app
    where you upload a video and see raw cuts vs. final ad-break
    recommendations on a timeline, plus tabs for every signal above.

## Why this approach

Ad-break placement in VOD is normally either fixed-interval (every N
minutes, regardless of content) or manual (an editor marks it up).
This tool automates finding *content-aware* candidates so ad breaks
land on natural pauses rather than arbitrary timestamps — and pairs
that with QC/brand-safety/engagement signals so the same pipeline that
finds ad breaks also flags what a human should check before the video
goes live.

Every signal in this project is deliberately grounded: sharpness is a
Laplacian-variance calculation you can verify by hand, loudness is an
RMS-based dBFS approximation with an explicit note on how it differs
from certified LUFS, mood congruence is labeled a heuristic proxy, and
brand safety is framed as a "needs review" flag rather than an
automated decision. Nothing here claims to predict viewer engagement
or ad performance — see each module's docstring for the specific
research or standard it's grounded in.

## Optional dependencies & graceful degradation

Two parts of the pipeline are optional and the app is written to
degrade gracefully if they're unavailable, per `src/app.py`:

- **YOLOv8 / `ultralytics`** — powers `object_ad_matcher.py` and
  `brand_safety.py`'s visual signal. If `ultralytics` isn't installed,
  those sections are skipped with an explanation instead of crashing
  the app.
- **Anthropic API key** — powers `script_reviewer.py`. If
  `ANTHROPIC_API_KEY` isn't set, that tab is skipped with an
  explanation; the rest of the pipeline runs fine without it.
- **Audio track** — if a video has no audio stream, or `ffmpeg`/
  `ffprobe` aren't on `PATH`, `audio_utils.extract_audio` raises
  `AudioExtractionError`, which `app.py` catches; every
  audio-dependent module then treats `audio_samples=None` as "skip
  this signal" rather than failing.

## Installation

```bash
pip install -r requirements.txt
```

You'll also need **ffmpeg + ffprobe** on `PATH` (not pip-installable —
used by `src/audio_utils.py` for audio extraction):

```bash
# Debian/Ubuntu
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

To enable the optional Script Generation + AI Reviewer tab, set an
Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running it

```bash
streamlit run src/app.py
```

Then upload a short (10–60s) MP4 with a few clear scene changes and,
ideally, an audio track (needed for the loudness, dead-air, and
audio-side brand-safety signals).

## Running tests

```bash
python3 tests/test_monetization.py
```

Tests cover the filtering/scoring logic in isolation from
PySceneDetect, using synthetic scene-cut data, so the monetization
logic can be verified independently of the video pipeline.

## Project structure

```
src/
  scene_detector.py     # PySceneDetect wrapper — raw scene cut detection
  monetization.py        # Filtering, scoring, ad-break windowing logic
  audio_utils.py          # ffmpeg-based audio extraction (shared)
  pacing_timeline.py      # Content Energy / Pacing Timeline
  production_quality.py   # Technical/Production Quality Score
  object_ad_matcher.py    # YOLOv8 contextual object → ad-category matching
  mood_congruence.py      # Mood Congruence Score
  brand_safety.py         # Brand Safety Flag (YOLOv8 + audio)
  dead_air_flag.py        # Dead-Air / Low-Engagement Flag
  script_reviewer.py      # Script Generation + AI Reviewer (optional, needs LLM)
  frame_features.py       # Shared per-frame feature extraction helpers
  app.py                  # Streamlit UI — wires all modules together
tests/
  test_monetization.py
sample_videos/
  synthetic_test.mp4     # Generated test clip with clear scene changes (no audio)
```

## Built with

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) —
  scene boundary detection
- OpenCV — video I/O backend for PySceneDetect and frame sampling
- [ffmpeg](https://ffmpeg.org/) — audio extraction (`src/audio_utils.py`)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) —
  object detection for contextual ad matching and brand safety
- Streamlit + Plotly — interactive UI and timeline visualization
- Anthropic API (optional) — script generation + AI review

## Possible extensions

- Certified LUFS loudness metering (e.g. via `pyloudnorm`) in place of
  the current RMS-based dBFS approximation, for delivery-grade
  compliance numbers.
- Batch processing across a folder of episodes with a summary report.
- Configurable "ad policies" (e.g. different fatigue/spacing rules per
  content genre).
- Calibrating the pacing/mood heuristics against real retention or
  engagement data, moving them from descriptive proxies toward
  validated predictors.