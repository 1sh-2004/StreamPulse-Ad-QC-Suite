"""
script_reviewer.py

Script Generation + AI Reviewer.

Two grounded, citable techniques combined into one feature:

  1. Video -> script: sample a keyframe at every scene-cut point and
     get a dense visual description of each from a vision-capable LLM.
     Stitched in order, this gives a scene-by-scene script of the
     whole video without a human watching it -- the same approach used
     by video-LLM pipelines (e.g. Qwen2.5-VL, VideoLLaMA2-style
     captioning), approximated cheaply via per-scene sampling instead
     of full video encoding.

  2. LLM-as-Judge, scoped honestly: the LLM reads the script (plus this
     pipeline's other module outputs) and produces a STRUCTURED
     QUALITATIVE critique against a fixed rubric -- pacing, tone
     consistency, narrative clarity, opening-hook strength, brand-
     safety notes in plain language. This mirrors the production
     pattern used by companies like OpusClip for short-form video
     curation. What it deliberately does NOT do is output a single
     numeric "will perform well" score -- that would require the kind
     of judge-vs-real-engagement validation loop those production
     systems run against their own data, which this pipeline has no
     way to replicate. The output here is an editor's-note-style
     report, not a rating.

This module only defines the interface and prompt construction --
it's decoupled from any specific LLM SDK so it can be wired up with
whichever vision/text API you have credentials for (Claude, GPT-4o,
etc.) via the `llm_call` and `vision_llm_call` callables passed in.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.scene_detector import SceneCut


@dataclass
class SceneDescription:
    scene_start_sec: float
    description: str


@dataclass
class ReviewerReport:
    """Raw structured critique text from the AI reviewer, plus the
    script it was generated from (kept together for display)."""
    script: List[SceneDescription]
    critique_text: str


# Callable signatures the caller is expected to provide:
#   frame_sampler(video_path, timestamp_sec) -> np.ndarray | None
#   vision_llm_call(frame, prompt) -> str          (one still frame -> description)
#   llm_call(prompt) -> str                        (text-only reviewer call)
FrameSampler = Callable
VisionLLMCall = Callable[..., str]
LLMCall = Callable[[str], str]


SCENE_DESCRIPTION_PROMPT = (
    "Describe this video frame in one dense sentence for a shot list: "
    "shot type (wide/medium/close-up), setting, main subject/action, "
    "and overall mood. Be concrete and specific, no filler."
)


def generate_scene_script(
    video_path: str,
    scene_cuts: List[SceneCut],
    frame_sampler: FrameSampler,
    vision_llm_call: VisionLLMCall,
) -> List[SceneDescription]:
    """
    For each scene cut, sample its keyframe and get a one-sentence
    dense description from a vision-capable LLM. Returns the ordered
    scene-by-scene script.
    """
    script = []
    for cut in scene_cuts:
        frame = frame_sampler(video_path, cut.timestamp_sec)
        if frame is None:
            continue
        description = vision_llm_call(frame, SCENE_DESCRIPTION_PROMPT)
        script.append(SceneDescription(scene_start_sec=cut.timestamp_sec, description=description.strip()))
    return script


REVIEWER_RUBRIC = """\
You are reviewing a scene-by-scene video script as a first-pass editor.
Score no numbers -- write plain-language notes only, organized under
these headings:

1. Pacing consistency -- does the energy/rhythm implied by the scene
   descriptions feel even, or are there jarring speed-ups/slow-downs?
2. Tone & mood shifts -- any abrupt tonal whiplash between adjacent scenes?
3. Narrative clarity -- does the sequence read as a coherent story/flow?
4. Opening hook strength -- does the first 1-2 scenes grab attention?
5. Brand-safety notes (plain language) -- anything in the descriptions
   that a brand-safety reviewer should double check.

Be specific and reference scene timestamps. Do not invent a numeric
score or a performance prediction -- this is qualitative editorial
feedback only.
"""


def build_ai_reviewer_prompt(
    script: List[SceneDescription],
    ad_value_notes: Optional[str] = None,
    brand_safety_notes: Optional[str] = None,
    pacing_notes: Optional[str] = None,
) -> str:
    """
    Assemble the full reviewer prompt: rubric + the generated script +
    optionally, plain-language summaries of this pipeline's other
    module outputs (ad-value windows, brand-safety flags, pacing
    timeline) so the reviewer can narrate the whole pipeline's
    findings as one coherent report instead of separate scores.
    """
    script_lines = "\n".join(
        f"[{s.scene_start_sec:.1f}s] {s.description}" for s in script
    )

    sections = [REVIEWER_RUBRIC, "\n## Scene-by-scene script\n", script_lines]

    if pacing_notes:
        sections.append(f"\n## Pacing/energy timeline (for context)\n{pacing_notes}")
    if ad_value_notes:
        sections.append(f"\n## Recommended ad-break windows (for context)\n{ad_value_notes}")
    if brand_safety_notes:
        sections.append(f"\n## Brand-safety flags detected (for context)\n{brand_safety_notes}")

    return "\n".join(sections)


def build_reviewer_report(
    script: List[SceneDescription],
    llm_call: LLMCall,
    ad_value_notes: Optional[str] = None,
    brand_safety_notes: Optional[str] = None,
    pacing_notes: Optional[str] = None,
) -> ReviewerReport:
    """
    Full entry point: build the prompt from the script (+ optional
    context from other modules) and call the text LLM to get the
    structured qualitative critique.
    """
    prompt = build_ai_reviewer_prompt(script, ad_value_notes, brand_safety_notes, pacing_notes)
    critique = llm_call(prompt)
    return ReviewerReport(script=script, critique_text=critique.strip())