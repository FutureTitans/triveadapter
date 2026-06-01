"""
analyzer.py — TRIBE v2 brain encoding inference logic.

Loads Meta's TRIBE v2 model, processes video/audio/text inputs,
and returns brain activation predictions with ROI scores.
"""

import os
import logging

# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Force CPU mode BEFORE importing torch or any ML library.
# The Docker image ships PyTorch compiled with CUDA stubs, which makes
# torch.cuda.is_available() return True even on CPU-only servers.
# This single line forces ALL downstream libraries (TRIBE v2, neuralset,
# transformers, whisperx, ctranslate2) to use CPU — eliminating every
# "Found no NVIDIA driver" crash in one shot.
# ══════════════════════════════════════════════════════════════════════════════
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model Singleton ──────────────────────────────────────────────────────────
_model = None
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _apply_cpu_patches():
    """
    Apply all monkey-patches needed to run TRIBE v2 on a CPU-only server.
    Called once before loading the model.
    """
    import torch
    import torchvision  # REQUIRED to register C++ operators like torchvision::nms!
    import tribev2.eventstransforms
    import neuralset.extractors.text

    # ── Patch 1: WhisperX compute_type ────────────────────────────────────
    # TRIBE v2 hardcodes compute_type="float16" for whisperx transcription.
    # float16 is only supported on CUDA. On CPU we must use "int8".
    original_get_transcript = (
        tribev2.eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio
    )

    @staticmethod
    def _patched_get_transcript(wav_filename, language="english"):
        import subprocess

        orig_run = subprocess.run

        def hooked_run(*args, **kwargs):
            if args and isinstance(args[0], list) and "whisperx" in args[0]:
                try:
                    idx = args[0].index("--compute_type")
                    args[0][idx + 1] = "int8"
                except ValueError:
                    pass
                # Also force device to cpu
                try:
                    idx = args[0].index("--device")
                    args[0][idx + 1] = "cpu"
                except ValueError:
                    pass
            return orig_run(*args, **kwargs)

        subprocess.run = hooked_run
        try:
            return original_get_transcript(wav_filename, language)
        finally:
            subprocess.run = orig_run

    tribev2.eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = (
        _patched_get_transcript
    )

    # ── Patch 2: HuggingFaceText model loading ────────────────────────────
    # neuralset's _load_model calls model.to(self.device) which crashes if
    # self.device resolved to "cuda" at class-definition time.
    # We completely replace _load_model for CPU to:
    #   a) Use bfloat16 to halve memory usage
    #   b) Skip the model.to("cuda") call entirely
    #   c) Handle all model types correctly

    def _patched_load_model(self, **kwargs):
        kwargs["low_cpu_mem_usage"] = True
        kwargs["torch_dtype"] = torch.bfloat16

        from transformers import AutoModel

        if "t5" in self.model_name or "bert" in self.model_name:
            from transformers import AutoModelForTextEncoding as Model
        elif "Phi-3" in self.model_name:
            from transformers import AutoModelForCausalLM as Model
        elif "Llama-3.2-11B-Vision" in self.model_name:
            from transformers import MllamaForConditionalGeneration as Model
        else:
            Model = AutoModel

        model = Model.from_pretrained(self.model_name, **kwargs)

        if not getattr(self, "pretrained", True):
            rawmodel = Model.from_config(model.config)
            with torch.no_grad():
                import itertools
                for p1, p2 in itertools.zip_longest(
                    model.parameters(), rawmodel.parameters()
                ):
                    p1.data = p2.to(p1)

        model.to("cpu")
        model.eval()
        return model

    neuralset.extractors.text.HuggingFaceText._load_model = _patched_load_model

    # ── Patch 3: HuggingFaceText.model property ──────────────────────────
    # The original property wraps exceptions in "Model loading went wrong"
    # which hides the real error. We replace it to print the full traceback.
    # We also use object.__setattr__ to bypass Pydantic's frozen model.

    @property
    def _patched_model_prop(self):
        if not hasattr(self, "_model"):
            from transformers import AutoTokenizer

            kwargs = {}
            if self.model_name.lower().startswith("microsoft/phi"):
                kwargs["trust_remote_code"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, truncation_side="left", **kwargs
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            # Use object.__setattr__ to bypass Pydantic freeze
            object.__setattr__(self, "_pad_id", self._tokenizer.eos_token_id)
            try:
                loaded = self._load_model()
                object.__setattr__(self, "_model", loaded)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Model loading error: {e}")
                raise RuntimeError(f"Model loading failed: {e}") from e
        return self._model

    neuralset.extractors.text.HuggingFaceText.model = _patched_model_prop

    logger.info("CPU patches applied successfully.")


def get_model():
    """
    Lazy-load the TRIBE v2 model. Caches to ./cache to avoid
    re-downloading the ~10GB weights on every restart.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from tribev2 import TribeModel

        # Apply all CPU compatibility patches
        _apply_cpu_patches()

        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info("Loading TRIBE v2 model (first load downloads ~10GB)...")
        _model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=CACHE_DIR,
        )
        logger.info("TRIBE v2 model loaded successfully.")
        return _model
    except Exception as e:
        logger.warning(f"TRIBE v2 model unavailable: {e}.")
        return None


# ── ROI Region Mapping (fsaverage5 vertex index ranges) ──────────────────────
# These are approximate vertex ranges for key brain regions on the fsaverage5
# mesh (~20,484 vertices). In production, use Nilearn's parcellation atlas.
ROI_REGIONS = {
    "visual_cortex":     (16000, 18000),  # Occipital lobe
    "auditory_cortex":   (10000, 12000),  # Superior temporal
    "language_areas":    (8000, 10000),   # Broca's + Wernicke's
    "attention_network": (4000, 6000),    # Dorsal parietal
    "emotion_centers":   (2000, 4000),    # Amygdala / insula region
    "memory_encoding":   (12000, 14000),  # Medial temporal
}


def _compute_roi_scores(brain_data: np.ndarray) -> dict:
    """Compute 0-100 scores for each ROI by averaging vertex activations."""
    roi_scores = {}
    global_min = brain_data.min()
    global_max = brain_data.max()
    spread = global_max - global_min if global_max != global_min else 1.0

    for region, (start, end) in ROI_REGIONS.items():
        end = min(end, len(brain_data))
        start = min(start, end)
        region_mean = brain_data[start:end].mean()
        # Normalize to 0-100
        score = float(((region_mean - global_min) / spread) * 100)
        roi_scores[region] = round(max(0, min(100, score)), 1)

    return roi_scores


def _compute_timeline(brain_data: np.ndarray, duration_seconds: int = 30) -> list:
    """
    Create a per-second engagement timeline by chunking the vertex array
    into `duration_seconds` segments and computing mean activation.
    """
    n_vertices = len(brain_data)
    chunk_size = max(1, n_vertices // duration_seconds)
    timeline = []

    for t in range(duration_seconds):
        start = t * chunk_size
        end = min(start + chunk_size, n_vertices)
        score = float(brain_data[start:end].mean())
        timeline.append({"time": t, "engagement_score": round(score, 4)})

    # Normalize scores to 0-100
    scores = [t["engagement_score"] for t in timeline]
    min_s, max_s = min(scores), max(scores)
    spread = max_s - min_s if max_s != min_s else 1.0
    for t in timeline:
        t["engagement_score"] = round(
            ((t["engagement_score"] - min_s) / spread) * 100, 1
        )

    return timeline


def _find_peak_moments(timeline: list, top_k: int = 5) -> list:
    """Find the top-K time indices with highest engagement scores."""
    sorted_moments = sorted(timeline, key=lambda x: x["engagement_score"], reverse=True)
    return sorted_moments[:top_k]


def _determine_dominant_modality(roi_scores: dict) -> str:
    """Determine which modality is most activated."""
    modality_map = {
        "visual": roi_scores.get("visual_cortex", 0),
        "auditory": roi_scores.get("auditory_cortex", 0),
        "language": roi_scores.get("language_areas", 0),
    }
    return max(modality_map, key=modality_map.get)


def _calculate_viral_score(roi_scores: dict) -> float:
    """
    Composite viral score: weighted average of ROI activations.
    Attention and emotion weigh more heavily for viral potential.
    """
    weights = {
        "visual_cortex": 0.15,
        "auditory_cortex": 0.10,
        "language_areas": 0.15,
        "attention_network": 0.25,
        "emotion_centers": 0.25,
        "memory_encoding": 0.10,
    }
    score = sum(roi_scores[k] * weights[k] for k in weights)
    return round(max(0, min(100, score)), 1)


def _generate_mock_brain_data() -> np.ndarray:
    """Generate realistic mock brain data when the real model isn't available."""
    np.random.seed(42)
    n_vertices = 20484  # fsaverage5

    # Start with a base pattern
    base = np.random.randn(n_vertices) * 0.3

    # Add regional hotspots for more realistic patterns
    for region, (start, end) in ROI_REGIONS.items():
        intensity = np.random.uniform(0.3, 1.0)
        base[start:end] += np.random.randn(end - start) * intensity + intensity

    return base


async def analyze_content(file_path: str = None, text: str = None) -> dict:
    """
    Main analysis pipeline. Accepts a file path (video/audio) or raw text.
    Returns brain activation data, ROI scores, timeline, and viral score.
    """
    model = get_model()
    preds_2d = None  # (n_timesteps, n_vertices) for real temporal data

    if model is not None:
        # ── Real TRIBE v2 inference ──────────────────────────────────────
        try:
            if file_path:
                ext = Path(file_path).suffix.lower()
                if ext in ('.mp4', '.mov'):
                    events_df = model.get_events_dataframe(video_path=file_path)
                elif ext in ('.mp3', '.wav'):
                    events_df = model.get_events_dataframe(audio_path=file_path)
                elif ext == '.txt':
                    events_df = model.get_events_dataframe(text_path=file_path)
                else:
                    events_df = model.get_events_dataframe(video_path=file_path)
                preds, segments = model.predict(events=events_df)
            elif text:
                # For text, create a temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    f.write(text)
                    tmp_path = f.name
                try:
                    events_df = model.get_events_dataframe(text_path=tmp_path)
                    preds, segments = model.predict(events=events_df)
                finally:
                    os.unlink(tmp_path)
            else:
                raise ValueError("No file or text provided")

            preds = np.array(preds, dtype=np.float64)

            # preds is (n_timesteps, n_vertices)
            if preds.ndim > 1:
                preds_2d = preds
                brain_data = preds.mean(axis=0)  # Average over time for spatial map
            else:
                brain_data = preds

        except Exception as e:
            logger.error(f"TRIBE v2 inference failed: {e}")
            raise RuntimeError(f"Brain analysis failed: {e}")
    else:
        # ── Model not available — raise error if HF_TOKEN is set ─────────
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            raise RuntimeError(
                "TRIBE v2 model could not be loaded despite HF_TOKEN being set. "
                "Check server logs for details."
            )
        raise RuntimeError(
            "TRIBE v2 model is not available. Set HF_TOKEN in environment to enable brain analysis."
        )

    # ── Compute derived metrics ──────────────────────────────────────────
    roi_scores = _compute_roi_scores(brain_data)

    # Use real temporal data if available, otherwise approximate from spatial
    if preds_2d is not None and preds_2d.shape[0] > 1:
        timeline = _compute_timeline_from_temporal(preds_2d)
    else:
        timeline = _compute_timeline(brain_data)

    peak_moments = _find_peak_moments(timeline)
    viral_score = _calculate_viral_score(roi_scores)
    dominant_modality = _determine_dominant_modality(roi_scores)

    return {
        "brain_data": brain_data.tolist(),
        "timeline": timeline,
        "peak_moments": peak_moments,
        "roi_scores": roi_scores,
        "viral_score": viral_score,
        "dominant_modality": dominant_modality,
    }


def _compute_timeline_from_temporal(preds_2d: np.ndarray) -> list:
    """
    Compute engagement timeline from real temporal predictions.
    preds_2d is (n_timesteps, n_vertices) — each row is one TR (~1-2 seconds).
    """
    # Mean activation across all vertices per timestep
    mean_per_timestep = preds_2d.mean(axis=1)

    # Normalize to 0-100
    min_s = mean_per_timestep.min()
    max_s = mean_per_timestep.max()
    spread = max_s - min_s if max_s != min_s else 1.0
    normalized = ((mean_per_timestep - min_s) / spread) * 100

    timeline = []
    for t, score in enumerate(normalized):
        timeline.append({"time": t, "engagement_score": round(float(score), 1)})

    return timeline
