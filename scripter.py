"""
scripter.py — Anthropic API integration for viral script generation.

Takes brain analysis results and generates hooks, scripts, social copy,
and tips using Claude as a viral content strategist.
"""

import os
import logging
import json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an elite viral content strategist with deep expertise in 
neuroscience-backed content creation. You have access to real brain activation data 
from a neuroscience model that predicts how audiences neurally respond to content.

Use the brain activation data provided to craft content that maximizes neural engagement.

Your responses must be in valid JSON format with exactly this structure:
{
  "hooks": ["hook1", "hook2", "hook3"],
  "script": "Full timestamped script...",
  "headline": "Viral headline",
  "caption": "Social media caption with hashtags",
  "tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}

Rules:
- Write 3 hook variations (each designed for the first 3 seconds)
- Write a full script with [00:00] timestamps
- Suggest b-roll/visual cuts at the peak brain activation moments
- Give a compelling headline and caption for social media
- Provide 5 actionable tips based on which brain regions were most activated
- If emotion_centers scored high, lean into emotional storytelling
- If attention_network scored high, use pattern interrupts and curiosity gaps
- If visual_cortex scored high, emphasize visual spectacle and dynamic cuts
- If auditory_cortex scored high, focus on sound design and voice modulation
- If language_areas scored high, use powerful word choices and rhetorical devices
- If memory_encoding scored high, use repetition and memorable phrases
"""


async def generate_script(
    roi_scores: dict,
    viral_score: float,
    peak_moments: list,
    dominant_modality: str,
    content_type: str = "Short-form",
    tone: str = "Educational",
) -> dict:
    """
    Generate a viral script using Anthropic's Claude API based on
    brain activation analysis results.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    user_prompt = f"""Here is the brain activation analysis for a piece of content:

**Viral Brain Score:** {viral_score}/100
**Dominant Neural Modality:** {dominant_modality}
**Content Type:** {content_type}
**Desired Tone:** {tone}

**Brain Region Activation Scores (0-100):**
- Visual Cortex: {roi_scores.get('visual_cortex', 0)}
- Auditory Cortex: {roi_scores.get('auditory_cortex', 0)}
- Language Areas: {roi_scores.get('language_areas', 0)}
- Attention Network: {roi_scores.get('attention_network', 0)}
- Emotion Centers: {roi_scores.get('emotion_centers', 0)}
- Memory Encoding: {roi_scores.get('memory_encoding', 0)}

**Peak Engagement Moments (seconds):** {json.dumps(peak_moments[:5])}

Based on this neural data, generate an optimized viral content package.
The content should be a {content_type} piece with a {tone} tone.
Maximize engagement by targeting the most activated brain regions.
Return ONLY valid JSON, no markdown code fences."""

    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response_text = message.content[0].text

            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(response_text[start:end])
                else:
                    raise ValueError("Could not parse JSON from Claude response")

            return result

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise RuntimeError(f"Script generation failed: {e}")
    else:
        logger.warning("ANTHROPIC_API_KEY not set. Using mock script generation.")
        return _generate_mock_script(roi_scores, viral_score, dominant_modality, content_type, tone)


def _generate_mock_script(
    roi_scores: dict,
    viral_score: float,
    dominant_modality: str,
    content_type: str,
    tone: str,
) -> dict:
    """Generate a realistic mock script when the API key isn't available."""

    top_region = max(roi_scores, key=roi_scores.get)
    region_labels = {
        "visual_cortex": "visual processing",
        "auditory_cortex": "auditory processing",
        "language_areas": "language comprehension",
        "attention_network": "sustained attention",
        "emotion_centers": "emotional response",
        "memory_encoding": "memory formation",
    }
    region_label = region_labels.get(top_region, "neural engagement")

    hooks = [
        f"🧠 Your brain just lit up — here's why this {content_type.lower()} will go viral...",
        f"The #1 trick neuroscientists use to hack {region_label} — and you can too.",
        f"Stop scrolling. Your {dominant_modality} cortex needs to see this.",
    ]

    script = f"""[00:00] HOOK — {hooks[0]}
[00:03] Open with a striking visual / pattern interrupt to engage the {dominant_modality} pathway.
[00:06] "Did you know your brain makes a decision about content in under 3 seconds?"
[00:10] Transition — introduce the core concept. Peak brain activation occurs here.
[00:15] Deep dive — explain the key insight with {tone.lower()} framing.
[00:20] B-ROLL: Show data visualization / brain scan imagery.
[00:25] Callback to hook — reinforce the emotional takeaway.
[00:28] CTA — "Follow for more neuroscience-backed content strategies."
[00:30] End card with subscribe/follow prompt."""

    headline = f"This {content_type} Hack Triggers {region_label.title()} — {viral_score}% Neural Engagement Score"

    caption = f"""🧠 We ran this through a neuroscience AI model and here's what happened...

Your brain's {dominant_modality} cortex lit up like a Christmas tree.
{region_label.title()} peaked at {roi_scores[top_region]}%.

This is the future of content creation — backed by real brain data.

#NeuroViral #ContentCreation #BrainScience #ViralContent #{dominant_modality.title()}"""

    tips = [
        f"Your {dominant_modality} pathway is strongest — lead with {dominant_modality} elements in the first 3 seconds.",
        f"Emotion centers scored {roi_scores.get('emotion_centers', 0)}% — {'amplify emotional hooks' if roi_scores.get('emotion_centers', 0) > 50 else 'add more emotional triggers'}.",
        f"Attention network at {roi_scores.get('attention_network', 0)}% — {'great sustained focus, keep pacing tight' if roi_scores.get('attention_network', 0) > 50 else 'add pattern interrupts every 5-7 seconds'}.",
        f"Memory encoding at {roi_scores.get('memory_encoding', 0)}% — {'content is memorable, good structure' if roi_scores.get('memory_encoding', 0) > 50 else 'use repetition and callbacks to boost recall'}.",
        f"Viral Brain Score: {viral_score}/100 — {'🔥 strong viral potential!' if viral_score > 70 else '⚡ optimize hook and emotional beats to push past 70.'}",
    ]

    return {
        "hooks": hooks,
        "script": script,
        "headline": headline,
        "caption": caption,
        "tips": tips,
    }
