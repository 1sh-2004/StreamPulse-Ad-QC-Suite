"""
app.py

Streamlit front-end for the Context-Aware Ad Insertion Engine.

Full pipeline (all ten feature modules), in run order:
  1. scene_detector.py      -- raw scene cuts (PySceneDetect)
  2. monetization.py        -- Ad Value Score: filter/score/space cuts
                                into ad-break windows
  3. audio_utils.py         -- extract audio track once, shared by
                                every audio-dependent module below
  4. pacing_timeline.py     -- Content Energy/Pacing Timeline
  5. production_quality.py  -- Technical/Production Quality Score
  6. object_ad_matcher.py   -- Contextual object detection -> ad
                                matching (YOLOv8)
  7. mood_congruence.py     -- Mood Congruence Score
  8. brand_safety.py        -- Brand Safety Flag (YOLOv8 + audio)
  9. dead_air_flag.py       -- Dead-Air/Low-Engagement Flag
 10. script_reviewer.py     -- Storyboard Generation & AI Reviewer
                                (Google Gemini Vision + Flash Judge)

Steps 6-10 need per-timestamp video frames; `sample_frame_at` below is
the shared `frame_sampler` callable those modules expect.

YOLOv8 (object_ad_matcher, brand_safety) and the AI reviewer
(script_reviewer) degrade gracefully if ultralytics is absent or the
Gemini API key is unconfigured/exhausted.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scene_detector import detect_scenes, get_video_duration, SceneCut
from src.monetization import build_ad_break_windows
from src.audio_utils import extract_audio, probe_audio_track, AudioExtractionError
from src.pacing_timeline import build_pacing_timeline, compute_motion_energy
from src.production_quality import (
    compute_sharpness,
    compute_exposure_stats,
    score_scene_quality,
    check_loudness_compliance,
    check_opening_hook_pacing,
)
from src.dead_air_flag import build_dead_air_flags

try:
    from src.object_ad_matcher import build_contextual_ad_matches, load_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from src.brand_safety import build_brand_safety_report
except ImportError:
    build_brand_safety_report = None

try:
    from src.mood_congruence import build_mood_congruence_report
except ImportError:
    build_mood_congruence_report = None

try:
    from src.script_reviewer import generate_scene_script, build_reviewer_report
except ImportError:
    generate_scene_script = None
    build_reviewer_report = None

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY"))
except ImportError:
    genai = None
    types = None
    GEMINI_AVAILABLE = False


st.set_page_config(page_title="Ad Insertion Engine", page_icon="🎬", layout="wide")


def sample_frame_at(video_path: str, timestamp_sec: float):
    """
    Shared frame_sampler callable: (video_path, timestamp_sec) -> BGR
    np.ndarray | None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


st.title("🎬 Context-Aware Ad Insertion Engine")
st.caption(
    "Upload a short video. We detect natural scene breaks (via PySceneDetect) "
    "and recommend ad-insertion points that avoid mid-action cuts and ad-fatigue."
)

with st.sidebar:
    st.header("Tuning parameters")
    threshold = st.slider(
        "Scene detection sensitivity", 10.0, 50.0, 27.0, step=1.0,
        help="Lower = more sensitive, detects more/smaller cuts."
    )
    fatigue_gap = st.slider(
        "Min gap between cuts (ad-fatigue filter, sec)", 1.0, 10.0, 3.0, step=0.5,
        help="Cuts closer together than this are collapsed -- avoids rapid-cut sequences."
    )
    edge_margin = st.slider(
        "Edge margin (sec)", 0.0, 30.0, 15.0, step=1.0,
        help="No ad breaks within this many seconds of start/end."
    )
    min_spacing = st.slider(
        "Min spacing between ad breaks (sec)", 10.0, 120.0, 45.0, step=5.0,
        help="Minimum time between two recommended ad breaks."
    )

uploaded_file = st.file_uploader("Upload a video (MP4 recommended)", type=["mp4", "mov", "mkv"])

if uploaded_file is not None:
    # Reset cached script/report when a new video file is uploaded
    if "current_video_name" not in st.session_state or st.session_state.current_video_name != uploaded_file.name:
        st.session_state.current_video_name = uploaded_file.name
        st.session_state.generated_script = None
        st.session_state.reviewer_report = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    with st.spinner("Analyzing video for scene breaks..."):
        raw_cuts = detect_scenes(video_path, threshold=threshold)
        duration = get_video_duration(video_path)

    if not raw_cuts:
        st.warning("No scene cuts detected. Try lowering the sensitivity threshold.")
    else:
        ad_windows = build_ad_break_windows(
            raw_cuts,
            video_duration_sec=duration,
            fatigue_gap_sec=fatigue_gap,
            edge_margin_sec=edge_margin,
            min_ad_spacing_sec=min_spacing,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Raw scene cuts detected", len(raw_cuts))
        col2.metric("Recommended ad breaks", len(ad_windows))
        col3.metric("Video duration", f"{duration:.1f}s")

        st.subheader("Timeline")

        fig = go.Figure()

        # Raw cuts as light tick marks along the bottom
        fig.add_trace(go.Scatter(
            x=[c.timestamp_sec for c in raw_cuts],
            y=[0.1] * len(raw_cuts),
            mode="markers",
            name="Raw scene cuts",
            marker=dict(symbol="line-ns", size=14, color="lightgray", line=dict(width=1)),
        ))

        # Ad break recommendations, colored by score
        if ad_windows:
            sorted_by_time = sorted(ad_windows, key=lambda w: w.timestamp_sec)
            fig.add_trace(go.Scatter(
                x=[w.timestamp_sec for w in sorted_by_time],
                y=[0.5] * len(sorted_by_time),
                mode="markers+text",
                name="Recommended ad breaks",
                marker=dict(
                    size=18,
                    color=[w.score for w in sorted_by_time],
                    colorscale="RdYlGn",
                    cmin=0, cmax=1,
                    line=dict(width=2, color="black"),
                    symbol="diamond",
                ),
                text=[f"{w.timestamp_sec:.1f}s" for w in sorted_by_time],
                textposition="top center",
            ))

        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis=dict(visible=False, range=[0, 1]),
            height=280,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Recommended ad-break windows (ranked by confidence)")
        if ad_windows:
            st.dataframe(
                [
                    {
                        "Timestamp (s)": round(w.timestamp_sec, 2),
                        "Confidence score": w.score,
                        "Reason": w.reason,
                    }
                    for w in ad_windows
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No ad-break windows survived the filters -- try loosening the spacing/edge settings.")

        # -------------------------------------------------------------
        # Shared prep: audio extraction + motion energy
        # -------------------------------------------------------------
        with st.spinner("Extracting audio track..."):
            try:
                audio_samples, audio_sample_rate = extract_audio(video_path)
            except AudioExtractionError as e:
                audio_samples, audio_sample_rate = None, None
                st.warning(f"Audio extraction unavailable: {e}")

        with st.spinner("Computing motion energy..."):
            motion_series = compute_motion_energy(video_path)

        pacing_points = build_pacing_timeline(
            raw_cuts, duration, motion_series, window_sec=10.0,
        )

        tabs = st.tabs([
            "⚡ Pacing", "🎚️ Production Quality", "🎯 Ad Matching",
            "🎭 Mood Congruence", "🚩 Brand Safety", "🔇 Dead Air",
            "📝 Storyboard", "⚖️ AI Reviewer",
        ])

        # --- Tab 0: Pacing / Content Energy Timeline -------------------
        with tabs[0]:
            st.caption(
                "Cut density + motion magnitude combined into a per-window "
                "energy score -- a descriptive pacing signal, not an "
                "engagement prediction (see pacing_timeline.py)."
            )
            if pacing_points:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=[p.timestamp_sec for p in pacing_points],
                    y=[p.energy_score for p in pacing_points],
                    mode="lines+markers",
                    name="Energy score",
                ))
                fig_p.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Energy (0-1)",
                    height=280, margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_p, width="stretch")
                st.dataframe(
                    [
                        {
                            "Time (s)": p.timestamp_sec, "Cut density": p.cut_density_score,
                            "Motion": p.motion_score, "Energy": p.energy_score, "Label": p.label,
                        }
                        for p in pacing_points
                    ],
                    width="stretch", hide_index=True,
                )
            else:
                st.info("Not enough data to build a pacing timeline.")

        # --- Tab 1: Production Quality Score ----------------------------
        with tabs[1]:
            st.caption(
                "Per-scene sharpness, exposure, and (video-wide) loudness "
                "compliance / opening-hook checks -- all formula-based QC "
                "signals (see production_quality.py)."
            )
            quality_rows = []
            for cut in raw_cuts:
                frame = sample_frame_at(video_path, cut.timestamp_sec)
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = compute_sharpness(gray)
                exposure = compute_exposure_stats(gray)
                overall, flags = score_scene_quality(
                    sharpness=sharpness, exposure=exposure,
                    stability_jitter=0.0, color_consistency_to_prev=None,
                )
                quality_rows.append({
                    "Time (s)": round(cut.timestamp_sec, 2),
                    "Sharpness": round(sharpness, 1),
                    "Blown highlights": exposure.pct_blown_highlights,
                    "Crushed blacks": exposure.pct_crushed_blacks,
                    "Quality score": overall,
                    "Flags": "; ".join(flags) if flags else "-",
                })
            if quality_rows:
                st.dataframe(quality_rows, width="stretch", hide_index=True)
            else:
                st.info("No frames could be sampled for quality scoring.")

            if audio_samples is not None:
                loudness = check_loudness_compliance(audio_samples)
                st.metric(
                    "Loudness (RMS dBFS, approx.)",
                    f"{loudness.measured_dbfs:.1f} dB",
                    delta=f"{loudness.delta_db:+.1f} dB vs. {loudness.target_dbfs} target",
                )
                st.caption(
                    "RMS approximation, not certified LUFS -- see "
                    "production_quality.py's note on ATSC A/85 / EBU R128."
                )
            hook = check_opening_hook_pacing(raw_cuts, motion_series)
            (st.success if not hook.flagged else st.warning)(hook.reason)

        # --- Tab 2: Contextual Object -> Ad Matching (YOLOv8) -----------
        scene_ad_matches = []
        with tabs[2]:
            st.caption(
                "YOLOv8 object detection per scene keyframe, mapped to "
                "advertising categories (see object_ad_matcher.py)."
            )
            if not YOLO_AVAILABLE:
                st.info("`ultralytics` is not installed -- run `pip install ultralytics` to enable this.")
            else:
                with st.spinner("Running YOLOv8 object detection..."):
                    try:
                        scene_ad_matches = build_contextual_ad_matches(
                            video_path, raw_cuts, sample_frame_at,
                        )
                    except Exception as e:
                        st.error(f"Object detection failed: {e}")
                rows = []
                for m in scene_ad_matches:
                    for match in m.matches:
                        rows.append({
                            "Time (s)": round(m.scene_start_sec, 2),
                            "Ad category": match.category,
                            "Confidence": match.confidence,
                            "Objects": ", ".join(match.supporting_objects),
                        })
                if rows:
                    st.dataframe(rows, width="stretch", hide_index=True)
                else:
                    st.info("No ad-relevant objects detected in sampled keyframes.")

        # --- Tab 3: Mood Congruence --------------------------------------
        with tabs[3]:
            st.caption(
                "Scene mood (from color/brightness + pacing energy) vs. "
                "each matched ad category's conventional mood -- see "
                "mood_congruence.py for the cited research."
            )
            if not YOLO_AVAILABLE:
                st.info("Requires object matching (enable `ultralytics` above) to know which ad category to check congruence against.")
            elif build_mood_congruence_report is None:
                st.info("mood_congruence.py could not be imported.")
            elif not scene_ad_matches:
                st.info("No ad-category matches yet -- see the Ad Matching tab.")
            else:
                mood_results = build_mood_congruence_report(
                    video_path, raw_cuts, sample_frame_at, scene_ad_matches, pacing_points,
                )
                if mood_results:
                    st.dataframe(
                        [
                            {
                                "Time (s)": round(r.scene_start_sec, 2), "Scene mood": r.scene_mood,
                                "Ad category": r.ad_category, "Expected mood": r.expected_mood,
                                "Congruence": r.congruence_score, "Reason": r.reason,
                            }
                            for r in mood_results
                        ],
                        width="stretch", hide_index=True,
                    )
                else:
                    st.info("No scenes with a matched ad category to score.")

        # --- Tab 4: Brand Safety Flag -------------------------------------
        safety_report = []
        flagged = []
        with tabs[4]:
            st.caption(
                "YOLOv8 for a narrow set of risky object classes + audio "
                "loudness-spike detection -- a 'needs human review' signal, "
                "not an auto block/allow (see brand_safety.py)."
            )
            if not YOLO_AVAILABLE or build_brand_safety_report is None:
                st.info("Requires `ultralytics` to be installed.")
            else:
                with st.spinner("Scanning for brand-safety signals..."):
                    try:
                        safety_report = build_brand_safety_report(
                            video_path, raw_cuts, sample_frame_at,
                            audio_samples=audio_samples, audio_sample_rate=audio_sample_rate,
                        )
                    except Exception as e:
                        st.error(f"Brand safety scan failed: {e}")
                flagged = [r for r in safety_report if r.flagged]
                st.metric("Scenes flagged for review", f"{len(flagged)} / {len(safety_report)}")
                if flagged:
                    st.dataframe(
                        [
                            {"Time (s)": round(r.scene_start_sec, 2), "Reason": r.reason}
                            for r in flagged
                        ],
                        width="stretch", hide_index=True,
                    )
                else:
                    st.success("No brand-safety signals triggered in this narrow check set.")

        # --- Tab 5: Dead-Air / Low-Engagement Flag -------------------------
        dead_air_flags = []
        with tabs[5]:
            st.caption(
                "Audio silence correlated with visual (pacing) energy -- "
                "silence alone isn't flagged as a problem, silence *and* "
                "low motion is (see dead_air_flag.py)."
            )
            if audio_samples is None:
                st.info("No audio track available to check for dead air.")
            else:
                dead_air_flags = build_dead_air_flags(
                    audio_samples, audio_sample_rate, pacing_points,
                )
                if dead_air_flags:
                    st.dataframe(
                        [
                            {
                                "Start (s)": f.start_sec, "End (s)": f.end_sec,
                                "Duration (s)": f.duration_sec, "Severity": f.severity,
                                "Reason": f.reason,
                            }
                            for f in dead_air_flags
                        ],
                        width="stretch", hide_index=True,
                    )
                else:
                    st.success("No dead-air windows detected.")

        # --- Tab 6: Storyboard / Script Generation --------------------------
        with tabs[6]:
            st.caption(
                "Per-scene descriptions from a vision model (Gemini 2.5 Flash), "
                "stitched into a chronological narrative storyboard."
            )
            if generate_scene_script is None:
                st.info("script_reviewer.py could not be imported.")
            elif not GEMINI_AVAILABLE:
                st.warning("⚠️ `GEMINI_API_KEY` is not set or has run out. Set the environment variable to enable scene-to-text generation.")
            else:
                if st.button("Generate Scene Storyboard"):
                    client = genai.Client()

                    def vision_llm_call(frame, prompt):
                        ok, buf = cv2.imencode(".jpg", frame)
                        if not ok:
                            return "(could not encode frame)"
                        try:
                            image_part = types.Part.from_bytes(
                                data=buf.tobytes(),
                                mime_type="image/jpeg"
                            )
                            resp = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[image_part, prompt]
                            )
                            return resp.text.strip()
                        except Exception as e:
                            err = str(e).lower()
                            if "quota" in err or "resource_exhausted" in err or "429" in err:
                                return "[Gemini API quota has run out / rate limit reached]"
                            elif "key" in err or "unauthenticated" in err or "credential" in err or "401" in err or "403" in err:
                                return "[Gemini API key is invalid or unauthorized]"
                            return f"[Error: {e}]"

                    with st.spinner("Analyzing scene keyframes with Gemini Vision..."):
                        st.session_state.generated_script = generate_scene_script(
                            video_path, raw_cuts, sample_frame_at, vision_llm_call
                        )

                if st.session_state.get("generated_script"):
                    st.subheader("Generated Storyboard")
                    for s in st.session_state.generated_script:
                        st.markdown(f"**[{s.scene_start_sec:.1f}s]** {s.description}")
                else:
                    st.info("👆 Click 'Generate Scene Storyboard' above to translate detected scenes into a text story.")

        # --- Tab 7: AI Reviewer (LLM-as-Judge) ------------------------------
        with tabs[7]:
            st.caption(
                "LLM-as-Judge review of the generated storyboard against qualitative "
                "storytelling, ad placement, and brand pacing rubrics."
            )
            if build_reviewer_report is None:
                st.info("script_reviewer.py could not be imported.")
            elif not GEMINI_AVAILABLE:
                st.warning("⚠️ `GEMINI_API_KEY` is not set or has run out.")
            elif not st.session_state.get("generated_script"):
                st.info("👆 Please go to the **📝 Storyboard** tab and generate the storyboard first before requesting an AI review.")
            else:
                if st.button("Run AI Reviewer"):
                    client = genai.Client()

                    def llm_call(prompt):
                        try:
                            resp = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt
                            )
                            return resp.text.strip()
                        except Exception as e:
                            err = str(e).lower()
                            if "quota" in err or "resource_exhausted" in err or "429" in err:
                                return "⚠️ Review could not be completed: The Gemini API quota has run out or the rate limit was reached."
                            elif "key" in err or "unauthenticated" in err or "credential" in err or "401" in err or "403" in err:
                                return "⚠️ Review could not be completed: The Gemini API key is invalid or unauthorized."
                            return f"⚠️ Review could not be completed: {e}"

                    ad_value_notes = "; ".join(
                        f"{w.timestamp_sec:.1f}s (score {w.score})" for w in ad_windows
                    ) or None

                    brand_notes = None
                    if flagged:
                        brand_notes = "; ".join(r.reason for r in flagged)

                    pacing_notes = "; ".join(
                        f"{p.timestamp_sec:.0f}s: {p.label}" for p in pacing_points
                    ) or None

                    with st.spinner("AI Reviewer judging storyboard against quality and brand rubrics..."):
                        report = build_reviewer_report(
                            st.session_state.generated_script,
                            llm_call,
                            ad_value_notes=ad_value_notes,
                            brand_safety_notes=brand_notes,
                            pacing_notes=pacing_notes,
                        )
                        st.session_state.reviewer_report = report

                if st.session_state.get("reviewer_report"):
                    st.subheader("AI Reviewer Report")
                    st.markdown(st.session_state.reviewer_report.critique_text)

    os.unlink(video_path)
else:
    st.info("👆 Upload a video to get started. Short clips (10-60s) with a few clear scene changes work best for demos.")