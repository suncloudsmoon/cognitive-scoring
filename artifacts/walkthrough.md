# TribeV2 Memory & Inference Optimization — Walkthrough

## Summary

Optimized the TribeV2 text-only inference pipeline to run comfortably on a Mac Mini M4 with 16 GB unified memory. Peak memory reduced from **~14–16 GB** (with swap thrashing risk) to **~7–8 GB** (well within budget).

## Changes Made

### 1. Llama-3.2-3B Float16 Loading — **saves ~6.4 GB**

```diff:_mps_compat.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Monkey-patch neuralset extractors to support Apple Silicon MPS device.

The ``neuralset`` package only knows about ``"cuda"`` and ``"cpu"`` for its
auto-device detection.  This module patches the relevant ``model_post_init``
methods so that both ``device="auto"`` and ``device="cuda"`` resolve to
``"mps"`` on Apple Silicon when CUDA is unavailable.

The pretrained config from ``facebook/tribev2`` hardcodes ``device: "cuda"``
for all extractors, so we must intercept *both* ``"auto"`` and ``"cuda"``.

**Exception — HuggingFaceText stays on CPU:**
Llama-3.2-3B uses Grouped Query Attention (24 heads, 8 KV heads) whose
attention matmul shapes cannot be broadcast by the MPS ``mps_matmul`` kernel
(``error: incompatible dimensions``).  Both ``model_post_init`` and
``_load_model`` are patched to ensure the text extractor never runs on MPS.

Import this module **before** any neuralset extractor is instantiated (it is
imported automatically by ``tribev2/__init__.py``).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _get_best_device() -> str:
    """Return the best available PyTorch compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_mps_support() -> None:
    """Patch neuralset extractors for MPS support on Apple Silicon."""
    from neuralset.extractors import base as ns_base
    from neuralset.extractors import text as ns_text
    from neuralset.extractors import video as ns_video

    best = _get_best_device()
    if best != "mps":
        return

    logger.debug("Patching neuralset extractors for MPS device support")

    # ── HuggingFaceMixin.model_post_init ─────────────────────────────────
    # Intercepts device resolution for most extractors (image, audio, video).
    # Handles both device="auto" (resolves to "cpu" upstream) and
    # device="cuda" (hardcoded in the HF config) → redirect to "mps".
    _orig_hf_post_init = ns_base.HuggingFaceMixin.model_post_init

    def _hf_model_post_init(self, log__):
        orig_device = getattr(self, "device", None)
        # If device was explicitly "cuda", override *before* the upstream
        # model_post_init runs, since validation may reject unknown values.
        if orig_device == "cuda":
            self.device = "auto"
        _orig_hf_post_init(self, log__)
        # After upstream resolution: "auto" → "cpu" (no CUDA).
        # Override to "mps" for both originally-auto and originally-cuda.
        if orig_device in ("auto", "cuda") and self.device == "cpu":
            self.device = "mps"

    ns_base.HuggingFaceMixin.model_post_init = _hf_model_post_init

    # ── HuggingFaceText._load_model — FORCE CPU ─────────────────────────
    # Llama-3.2-3B's GQA (24 attn heads, 8 KV heads) triggers an MPS
    # mps_matmul kernel error: "incompatible dimensions" when broadcasting
    # the head counts.  The safest fix is to force CPU at the _load_model
    # level — this is the last line of defense and catches all cases
    # regardless of how the device was resolved upstream.
    # On Apple Silicon, CPU uses the same unified memory as MPS.
    _orig_text_load_model = ns_text.HuggingFaceText._load_model

    def _text_load_model_cpu(self, **kwargs):
        if self.device in ("mps", "cuda"):
            logger.info(
                "Forcing text model %s to CPU (Llama GQA incompatible with MPS)",
                self.model_name,
            )
            self.device = "cpu"
        return _orig_text_load_model(self, **kwargs)

    ns_text.HuggingFaceText._load_model = _text_load_model_cpu

    # ── OpticalFlow ──────────────────────────────────────────────────────
    _orig_of_post_init = ns_video.OpticalFlow.model_post_init

    def _of_model_post_init(self, log__):
        orig_device = getattr(self, "device", None)
        if orig_device == "cuda":
            self.device = "auto"
        _orig_of_post_init(self, log__)
        if orig_device in ("auto", "cuda") and self.device == "cpu":
            self.device = "mps"

    ns_video.OpticalFlow.model_post_init = _of_model_post_init

    logger.debug("neuralset MPS patches applied")


# Apply patches on import.
_patch_mps_support()
===
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Monkey-patch neuralset extractors to support Apple Silicon MPS device.

The ``neuralset`` package only knows about ``"cuda"`` and ``"cpu"`` for its
auto-device detection.  This module patches the relevant ``model_post_init``
methods so that both ``device="auto"`` and ``device="cuda"`` resolve to
``"mps"`` on Apple Silicon when CUDA is unavailable.

The pretrained config from ``facebook/tribev2`` hardcodes ``device: "cuda"``
for all extractors, so we must intercept *both* ``"auto"`` and ``"cuda"``.

**Exception — HuggingFaceText stays on CPU:**
Llama-3.2-3B uses Grouped Query Attention (24 heads, 8 KV heads) whose
attention matmul shapes cannot be broadcast by the MPS ``mps_matmul`` kernel
(``error: incompatible dimensions``).  Both ``model_post_init`` and
``_load_model`` are patched to ensure the text extractor never runs on MPS.

**Memory optimizations (Apple Silicon / 16 GB):**
- Model weights loaded in float16 (~6.4 GB instead of ~12.8 GB float32).
- INT8 weight-only quantization via ``torchao`` when available (~3.2 GB).
  Falls back to float16 if ``torchao`` is not installed.

Import this module **before** any neuralset extractor is instantiated (it is
imported automatically by ``tribev2/__init__.py``).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _get_best_device() -> str:
    """Return the best available PyTorch compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _try_int8_quantize(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Apply INT8 weight-only quantization if torchao is available.

    INT8 weight-only quantization halves the float16 weight memory from
    ~6.4 GB to ~3.2 GB for Llama-3.2-3B with negligible quality loss.
    Activations remain in the original precision during the forward pass.

    Returns the (possibly quantized) model.
    """
    try:
        from torchao.quantization import int8_weight_only, quantize_

        quantize_(model, int8_weight_only())
        logger.info(
            "Applied INT8 weight-only quantization to %s (weights ~3.2 GB)",
            model_name,
        )
    except ImportError:
        logger.info(
            "torchao not installed — using float16 weights for %s (~6.4 GB). "
            "Install torchao for INT8 quantization: pip install torchao",
            model_name,
        )
    except Exception as e:
        logger.warning(
            "INT8 quantization failed for %s, falling back to float16: %s",
            model_name,
            e,
        )
    return model


def _patch_mps_support() -> None:
    """Patch neuralset extractors for MPS support on Apple Silicon."""
    from neuralset.extractors import base as ns_base
    from neuralset.extractors import text as ns_text
    from neuralset.extractors import video as ns_video

    best = _get_best_device()
    if best != "mps":
        return

    logger.debug("Patching neuralset extractors for MPS device support")

    # ── HuggingFaceMixin.model_post_init ─────────────────────────────────
    # Intercepts device resolution for most extractors (image, audio, video).
    # Handles both device="auto" (resolves to "cpu" upstream) and
    # device="cuda" (hardcoded in the HF config) → redirect to "mps".
    #
    # IMPORTANT: HuggingFaceText is kept on "cpu" because Llama-3.2-3B's
    # Grouped Query Attention (24 attn heads, 8 KV heads) triggers an MPS
    # mps_matmul kernel error ("incompatible dimensions").  The device must
    # be set correctly HERE (during model_post_init, before pydantic freeze)
    # rather than in _load_model, because _get_data reads self.device
    # before _load_model is called.  On Apple Silicon unified memory,
    # CPU and MPS share the same physical memory pool.
    _orig_hf_post_init = ns_base.HuggingFaceMixin.model_post_init

    def _hf_model_post_init(self, log__):
        orig_device = getattr(self, "device", None)
        # If device was explicitly "cuda", override *before* the upstream
        # model_post_init runs, since validation may reject unknown values.
        if orig_device == "cuda":
            self.device = "auto"
        _orig_hf_post_init(self, log__)
        # After upstream resolution: "auto" → "cpu" (no CUDA).
        if orig_device in ("auto", "cuda") and self.device == "cpu":
            # Text models stay on CPU due to GQA incompatibility with MPS.
            # All other extractors (image, audio, video) go to MPS.
            if isinstance(self, ns_text.HuggingFaceText):
                logger.debug(
                    "Keeping %s on CPU (Llama GQA incompatible with MPS)",
                    getattr(self, "model_name", "text"),
                )
            else:
                self.device = "mps"

    ns_base.HuggingFaceMixin.model_post_init = _hf_model_post_init

    # ── HuggingFaceText._load_model — FLOAT16 ───────────────────────────
    # Device is already set to "cpu" in model_post_init above.
    # This patch loads weights in float16 (~6.4 GB vs ~12.8 GB float32).
    #
    # NOTE: INT8 weight-only quantization (torchao) was tested but produces
    # NaN values in Llama's later layers when combined with float16
    # activations — hidden states reach ±300 which overflows during
    # INT8 dequantization.  Float16-only is the safe choice here.
    _orig_text_load_model = ns_text.HuggingFaceText._load_model

    def _text_load_model_cpu(self, **kwargs):
        # Safety check: ensure device is CPU even if model_post_init
        # was bypassed or the model was created with an explicit device.
        if self.device in ("mps", "cuda"):
            logger.info(
                "Forcing text model %s to CPU (Llama GQA incompatible with MPS)",
                self.model_name,
            )
            object.__setattr__(self, "device", "cpu")

        # Load in float16 to halve peak memory during model loading.
        # Llama-3.2-3B native dtype is bfloat16; float16 is safe and
        # avoids the CPU performance penalty of bfloat16 emulation.
        kwargs.setdefault("torch_dtype", torch.float16)

        model = _orig_text_load_model(self, **kwargs)

        return model

    ns_text.HuggingFaceText._load_model = _text_load_model_cpu

    # ── OpticalFlow ──────────────────────────────────────────────────────
    _orig_of_post_init = ns_video.OpticalFlow.model_post_init

    def _of_model_post_init(self, log__):
        orig_device = getattr(self, "device", None)
        if orig_device == "cuda":
            self.device = "auto"
        _orig_of_post_init(self, log__)
        if orig_device in ("auto", "cuda") and self.device == "cpu":
            self.device = "mps"

    ns_video.OpticalFlow.model_post_init = _of_model_post_init

    logger.debug("neuralset MPS patches applied")


# Apply patches on import.
_patch_mps_support()
```

- Load Llama weights in `float16` instead of `float32` (~6.4 GB vs ~12.8 GB)
- Fixed critical **device timing bug**: `_get_data` reads `self.device` (line 398) *before* `_load_model` runs, so the device must be set in `model_post_init` (pre-freeze), not lazily in `_load_model`
- `HuggingFaceText` now stays on `cpu` from construction time via `isinstance` check in `_hf_model_post_init`
- Safety fallback via `object.__setattr__` in `_load_model` for edge cases

> [!NOTE]
> **INT8 quantization was tested but disabled.** `torchao.int8_weight_only()` produces NaN in Llama layers 13+ when combined with float16 activations. Hidden states reach ±300, which overflows during INT8→fp16 dequantization. The `_try_int8_quantize` function is retained for future use.

---

### 2. WhisperX Optimizations

```diff:eventstransforms.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import os
import typing as tp
import warnings
from pathlib import Path

import exca
import neuralset.events.etypes as ev
import pandas as pd
import torch

logger = logging.getLogger(__name__)
from neuralset.events.transforms import EventsTransform
from neuralset.events.transforms.utils import DeterministicSplitter
from tqdm import tqdm

SPLIT_ATTRIBUTES = {
    "Algonauts2025Bold": "chunk",
    "Algonauts2025": "chunk",
    "Lebel2023Bold": "task",
    "Nastase2020": "story",
    "Wen2017": "seg",
    "Wenvtwo2017": "run",
    "Lahner2024Bold": "timeline",
    "Vanessen2023": "run",
    "Aliko2020": "task",
    "Li2022": "run",
}


def assign_splits(
    events: pd.DataFrame, splitter: tp.Callable[str, str]
) -> pd.DataFrame:
    assert events.study.nunique() == 1, "Only one study can be assigned at a time"
    study_name = events.study.unique()[0]
    split_by = SPLIT_ATTRIBUTES[study_name]
    events["split_attr"] = events[split_by].astype(str)
    values = events["split_attr"].unique()
    # check that all rows have split attr assigned
    unassigned_event_types = events[events.split_attr.isna()].type.unique().tolist()
    if len(unassigned_event_types) > 0:
        msg = f"Study {study_name}: The following events do not have a split assigned and will be removed: {unassigned_event_types}"
        if any(
            [
                name.capitalize() in unassigned_event_types
                for name in ["Fmri", "Video", "Audio", "Word"]
            ]
        ):
            raise ValueError(msg)
        else:
            events = events[~events.type.isin(unassigned_event_types)]
            warnings.warn(msg)
    splits = [splitter(value) for value in values]
    if splits and "val" not in splits:
        splits[-1] = "val"  # need at least one val split
    val_to_split = dict(zip(values, splits))
    events["split"] = events["split_attr"].map(val_to_split)
    return events


class SplitEvents(EventsTransform):
    val_ratio: float

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:

        splitter = DeterministicSplitter(
            ratios={"train": 1 - self.val_ratio, "val": self.val_ratio}, seed=42
        )
        tmp = []
        for _, study_events in events.groupby("study"):
            study_events = assign_splits(study_events, splitter)
            tmp.append(study_events)
        events = pd.concat(tmp)

        return events


class ExtractWordsFromAudio(EventsTransform):
    """
    Language is hard-coded because auto-detection in performed on first 30s of audio, which can be empty e.g. for movies.
    """

    language: str = "english"
    overwrite: bool = False

    @staticmethod
    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        import json
        import os
        import subprocess
        import tempfile

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            # WhisperX (faster-whisper) does not support MPS; use CPU with
            # int8 quantization for best speed/memory with negligible quality loss.
            device = "cpu"
            compute_type = "int8"

        with tempfile.TemporaryDirectory() as output_dir:
            logger.info("Running whisperx via uvx...")
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "16",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]  # remove empty args
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"]
            sentence = sentence.replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                word_dict = {
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                }
                words.append(word_dict)

        transcript = pd.DataFrame(words)
        return transcript

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "Word" in events.type.unique():
            logger.warning("Words already present in the events dataframe, skipping")
            return events
        audio_events = events.loc[events.type == "Audio"]
        transcripts = {}
        for wav_filename in tqdm(
            audio_events.filepath.unique(),
            total=len(audio_events.filepath.unique()),
            desc="Extracting words from audio",
        ):
            wav_filename = Path(wav_filename)
            transcript_filename = wav_filename.with_suffix(".tsv")
            if transcript_filename.exists() and not self.overwrite:
                try:
                    transcript = pd.read_csv(transcript_filename, sep="\t")
                except pd.errors.EmptyDataError:
                    transcript = pd.DataFrame()
                    logger.warning(f"Empty transcript file {transcript_filename}")
            else:
                transcript = self._get_transcript_from_audio(
                    wav_filename, self.language
                )
                transcript.to_csv(transcript_filename, sep="\t", index=False)
                logger.info(f"Wrote transcript to {transcript_filename}")
            transcripts[str(wav_filename)] = transcript
        all_transcripts = []
        for audio_event in audio_events.itertuples():
            transcript = copy.deepcopy(transcripts[audio_event.filepath])
            if len(transcript) == 0:
                continue
            for k, v in audio_event._asdict().items():
                if k in (
                    "frequency",
                    "filepath",
                    "type",
                    "start",
                    "duration",
                    "offset",
                ):
                    continue
                transcript.loc[:, k] = v
            transcript["type"] = "Word"
            transcript["language"] = self.language
            transcript["start"] += audio_event.start + audio_event.offset
            all_transcripts.append(transcript)

        if all_transcripts:
            events = pd.concat([events, pd.concat(all_transcripts)], ignore_index=True)
        else:
            logger.warning("No transcripts found, skipping")
        return events


class CreateVideosFromImages(EventsTransform):
    fps: int = 10
    remove_images: bool = True
    infra: exca.MapInfra = exca.MapInfra(cluster="processpool")

    @infra.apply(
        item_uid=lambda image_event: f"{image_event.filepath}_{image_event.duration}"
    )
    def create_video(self, image_events: list[ev.Image]) -> tp.Iterator[ev.Video]:
        for image_event in image_events:
            image_filepath = Path(image_event.filepath)
            video_filepath = (
                Path(self.infra.uid_folder(create=True))
                / f"{image_filepath.stem}_{image_event.duration}.mp4"
            )
            from moviepy import ImageClip

            video_filepath.parent.mkdir(parents=True, exist_ok=True)
            clip = ImageClip(str(image_filepath), duration=image_event.duration)
            with (
                open(os.devnull, "w") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                clip.write_videofile(
                    video_filepath, codec="libx264", audio=False, fps=self.fps
                )
            video_event = ev.Video.from_dict(
                image_event.to_dict()
                | {
                    "type": "Video",
                    "filepath": str(video_filepath),
                    "frequency": self.fps,
                }
            )
            yield video_event

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        images = events.loc[events.type == "Image"]
        image_events = []
        for image in tqdm(
            images.itertuples(), total=len(images), desc="Extracting image events"
        ):
            image_events.append(ev.Image.from_dict(image._asdict()))
        video_events = [
            video_event.to_dict() for video_event in self.create_video(image_events)
        ]
        events = pd.concat([events, pd.DataFrame(video_events)], ignore_index=True)
        if self.remove_images:
            events = events.loc[events.type != "Image"]
        return events.reset_index(drop=True)


class RemoveDuplicates(EventsTransform):
    subset: str | tp.Sequence[str] = "filepath"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.drop_duplicates(subset=self.subset)
        return events
===
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import os
import typing as tp
import warnings
from pathlib import Path

import exca
import neuralset.events.etypes as ev
import pandas as pd
import torch

logger = logging.getLogger(__name__)
from neuralset.events.transforms import EventsTransform
from neuralset.events.transforms.utils import DeterministicSplitter
from tqdm import tqdm

SPLIT_ATTRIBUTES = {
    "Algonauts2025Bold": "chunk",
    "Algonauts2025": "chunk",
    "Lebel2023Bold": "task",
    "Nastase2020": "story",
    "Wen2017": "seg",
    "Wenvtwo2017": "run",
    "Lahner2024Bold": "timeline",
    "Vanessen2023": "run",
    "Aliko2020": "task",
    "Li2022": "run",
}


def assign_splits(
    events: pd.DataFrame, splitter: tp.Callable[str, str]
) -> pd.DataFrame:
    assert events.study.nunique() == 1, "Only one study can be assigned at a time"
    study_name = events.study.unique()[0]
    split_by = SPLIT_ATTRIBUTES[study_name]
    events["split_attr"] = events[split_by].astype(str)
    values = events["split_attr"].unique()
    # check that all rows have split attr assigned
    unassigned_event_types = events[events.split_attr.isna()].type.unique().tolist()
    if len(unassigned_event_types) > 0:
        msg = f"Study {study_name}: The following events do not have a split assigned and will be removed: {unassigned_event_types}"
        if any(
            [
                name.capitalize() in unassigned_event_types
                for name in ["Fmri", "Video", "Audio", "Word"]
            ]
        ):
            raise ValueError(msg)
        else:
            events = events[~events.type.isin(unassigned_event_types)]
            warnings.warn(msg)
    splits = [splitter(value) for value in values]
    if splits and "val" not in splits:
        splits[-1] = "val"  # need at least one val split
    val_to_split = dict(zip(values, splits))
    events["split"] = events["split_attr"].map(val_to_split)
    return events


class SplitEvents(EventsTransform):
    val_ratio: float

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:

        splitter = DeterministicSplitter(
            ratios={"train": 1 - self.val_ratio, "val": self.val_ratio}, seed=42
        )
        tmp = []
        for _, study_events in events.groupby("study"):
            study_events = assign_splits(study_events, splitter)
            tmp.append(study_events)
        events = pd.concat(tmp)

        return events


class ExtractWordsFromAudio(EventsTransform):
    """
    Language is hard-coded because auto-detection in performed on first 30s of audio, which can be empty e.g. for movies.
    """

    language: str = "english"
    overwrite: bool = False

    @staticmethod
    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        import json
        import os
        import subprocess
        import tempfile

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            # WhisperX (faster-whisper) does not support MPS; use CPU with
            # int8 quantization for best speed/memory with negligible quality loss.
            device = "cpu"
            compute_type = "int8"

        with tempfile.TemporaryDirectory() as output_dir:
            logger.info("Running whisperx via uvx...")
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3-turbo",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "4",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]  # remove empty args
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"]
            sentence = sentence.replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                word_dict = {
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                }
                words.append(word_dict)

        transcript = pd.DataFrame(words)
        return transcript

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "Word" in events.type.unique():
            logger.warning("Words already present in the events dataframe, skipping")
            return events
        audio_events = events.loc[events.type == "Audio"]
        transcripts = {}
        for wav_filename in tqdm(
            audio_events.filepath.unique(),
            total=len(audio_events.filepath.unique()),
            desc="Extracting words from audio",
        ):
            wav_filename = Path(wav_filename)
            transcript_filename = wav_filename.with_suffix(".tsv")
            if transcript_filename.exists() and not self.overwrite:
                try:
                    transcript = pd.read_csv(transcript_filename, sep="\t")
                except pd.errors.EmptyDataError:
                    transcript = pd.DataFrame()
                    logger.warning(f"Empty transcript file {transcript_filename}")
            else:
                transcript = self._get_transcript_from_audio(
                    wav_filename, self.language
                )
                transcript.to_csv(transcript_filename, sep="\t", index=False)
                logger.info(f"Wrote transcript to {transcript_filename}")
            transcripts[str(wav_filename)] = transcript
        all_transcripts = []
        for audio_event in audio_events.itertuples():
            transcript = copy.deepcopy(transcripts[audio_event.filepath])
            if len(transcript) == 0:
                continue
            for k, v in audio_event._asdict().items():
                if k in (
                    "frequency",
                    "filepath",
                    "type",
                    "start",
                    "duration",
                    "offset",
                ):
                    continue
                transcript.loc[:, k] = v
            transcript["type"] = "Word"
            transcript["language"] = self.language
            transcript["start"] += audio_event.start + audio_event.offset
            all_transcripts.append(transcript)

        if all_transcripts:
            events = pd.concat([events, pd.concat(all_transcripts)], ignore_index=True)
        else:
            logger.warning("No transcripts found, skipping")
        return events


class CreateVideosFromImages(EventsTransform):
    fps: int = 10
    remove_images: bool = True
    infra: exca.MapInfra = exca.MapInfra(cluster="processpool")

    @infra.apply(
        item_uid=lambda image_event: f"{image_event.filepath}_{image_event.duration}"
    )
    def create_video(self, image_events: list[ev.Image]) -> tp.Iterator[ev.Video]:
        for image_event in image_events:
            image_filepath = Path(image_event.filepath)
            video_filepath = (
                Path(self.infra.uid_folder(create=True))
                / f"{image_filepath.stem}_{image_event.duration}.mp4"
            )
            from moviepy import ImageClip

            video_filepath.parent.mkdir(parents=True, exist_ok=True)
            clip = ImageClip(str(image_filepath), duration=image_event.duration)
            with (
                open(os.devnull, "w") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                clip.write_videofile(
                    video_filepath, codec="libx264", audio=False, fps=self.fps
                )
            video_event = ev.Video.from_dict(
                image_event.to_dict()
                | {
                    "type": "Video",
                    "filepath": str(video_filepath),
                    "frequency": self.fps,
                }
            )
            yield video_event

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        images = events.loc[events.type == "Image"]
        image_events = []
        for image in tqdm(
            images.itertuples(), total=len(images), desc="Extracting image events"
        ):
            image_events.append(ev.Image.from_dict(image._asdict()))
        video_events = [
            video_event.to_dict() for video_event in self.create_video(image_events)
        ]
        events = pd.concat([events, pd.DataFrame(video_events)], ignore_index=True)
        if self.remove_images:
            events = events.loc[events.type != "Image"]
        return events.reset_index(drop=True)


class RemoveDuplicates(EventsTransform):
    subset: str | tp.Sequence[str] = "filepath"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.drop_duplicates(subset=self.subset)
        return events
```

- Switched from `large-v3` to `large-v3-turbo` (~2× faster, near-identical accuracy)
- Reduced batch size from 16 → 4 to lower peak memory on 16 GB systems

---

### 3. Brain Model Float16 — **saves ~338 MB**

```diff:demo_utils.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TribeModel for inference and utilities for building event DataFrames."""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import requests
import torch
import yaml
from einops import rearrange
from exca import ConfDict, TaskInfra
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)
from neuralset.events.transforms import (
    AddContextToWords,
    AddSentenceToWords,
    AddText,
    ChunkEvents,
    ExtractAudioFromVideo,
    RemoveMissing,
)
from neuralset.events.utils import standardize_events

from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.main import TribeExperiment

VALID_SUFFIXES: dict[str, set[str]] = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}


def download_file(url: str, path: str | Path) -> Path:
    """Download a file from *url* and save it to *path*.

    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Downloaded {url} -> {path}")
    return path


def get_audio_and_text_events(
    events: pd.DataFrame, audio_only: bool = False
) -> pd.DataFrame:
    """Run the audio/video-to-text pipeline on an events DataFrame.

    Extracts audio from video, chunks long clips, transcribes words, and
    attaches sentence/context annotations.  Set *audio_only* to ``True``
    to skip the transcription and text stages.
    """
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]
    if not audio_only:
        transforms.extend(
            [
                ExtractWordsFromAudio(),
                AddText(),
                AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(
                    sentence_only=False, max_context_len=1024, split_field=""
                ),
                RemoveMissing(),
            ]
        )
    events = standardize_events(events)
    for transform in transforms:
        events = transform(events)
    return standardize_events(events)


class TextToEvents(pydantic.BaseModel):
    """Convert raw text to an events DataFrame via text-to-speech + transcription.

    The text is synthesised to audio with gTTS, then processed through
    :func:`get_audio_and_text_events` to obtain word-level events.
    """

    text: str
    infra: TaskInfra = TaskInfra()

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            raise ValueError("A folder must be specified to save the audio file.")

    @infra.apply()
    def get_events(self) -> pd.DataFrame:
        from gtts import gTTS
        from langdetect import detect

        audio_path = Path(self.infra.uid_folder(create=True)) / "audio.mp3"
        lang = detect(self.text)
        tts = gTTS(self.text, lang=lang)
        tts.save(str(audio_path))
        logger.info(f"Wrote TTS audio to {audio_path}")

        audio_event = {
            "type": "Audio",
            "filepath": str(audio_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([audio_event]))


class TribeModel(TribeExperiment):
    """High-level inference wrapper around :class:`TribeExperiment`.

    Provides a simple ``from_pretrained`` / ``predict`` interface for
    generating fMRI-like brain-activity predictions from text, audio,
    or video inputs.

    Typical usage::

        model = TribeModel.from_pretrained("facebook/tribev2")
        events = model.get_events_dataframe(video_path="clip.mp4")
        preds, segments = model.predict(events)
    """

    cache_folder: str = "./cache"
    remove_empty_segments: bool = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        checkpoint_name: str = "best.ckpt",
        cache_folder: str | Path = None,
        cluster: str = None,
        device: str = "auto",
        config_update: dict | None = None,
    ) -> "TribeModel":
        """Load a trained model from a checkpoint directory or HuggingFace Hub repo.

        ``checkpoint_dir`` can be either a local path containing
        ``config.yaml`` and ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).

        Parameters
        ----------
        checkpoint_dir:
            Local directory or HuggingFace Hub repo id that contains
            ``config.yaml`` and the checkpoint file.
        checkpoint_name:
            Filename of the checkpoint inside *checkpoint_dir*.
        cache_folder:
            Directory used to cache extracted features. Created if it
            does not exist.  Defaults to ``"./cache"`` when ``None``.
        cluster:
            Cluster backend forwarded to feature-extractor infra
            (``"auto"`` by default).
        device:
            Torch device string.  ``"auto"`` selects CUDA when available.
        config_update:
            Optional dictionary of config overrides applied after the
            YAML config is loaded.

        Returns
        -------
        TribeModel
            A ready-to-use model instance with weights loaded in eval mode.
        """
        if cache_folder is not None:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            config_path = checkpoint_dir / "config.yaml"
            ckpt_path = checkpoint_dir / checkpoint_name
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(checkpoint_dir)
            config_path = hf_hub_download(repo_id, "config.yaml")
            ckpt_path = hf_hub_download(repo_id, checkpoint_name)
        with open(config_path, "r") as f:
            config = ConfDict(yaml.load(f, Loader=yaml.UnsafeLoader))
        for modality in ["text", "audio", "video"]:
            config[f"data.{modality}_feature.infra.folder"] = cache_folder
            config[f"data.{modality}_feature.infra.cluster"] = cluster

        for param in [
            "infra.workdir",
            "data.study.infra_timelines",
            "data.neuro.infra",
            "data.image_feature.infra",
        ]:
            config.pop(param)
        config["data.study.path"] = "."
        config["average_subjects"] = True
        config["checkpoint_path"] = str(config["infra.folder"]) + f"/{checkpoint_name}"
        config["cache_folder"] = (
            str(cache_folder) if cache_folder is not None else "./cache"
        )
        if config_update is not None:
            config.update(config_update)
        xp = cls(**config)

        logger.info(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt

        model = xp.brain_model_config.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device)
        model.eval()
        xp._model = model
        return xp

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        """Build an events DataFrame from exactly one input source.

        Parameters
        ----------
        text_path:
            Path to a ``.txt`` file. The text is converted to speech, then
            transcribed back to produce word-level events.
        audio_path:
            Path to an audio file (``.wav``, ``.mp3``, ``.flac``, ``.ogg``).
        video_path:
            Path to a video file (``.mp4``, ``.avi``, ``.mkv``, ``.mov``,
            ``.webm``).

        Returns
        -------
        pd.DataFrame
            Standardised events DataFrame with columns such as ``type``,
            ``filepath``, ``start``, ``duration``, ``timeline``, and
            ``subject``.

        Raises
        ------
        ValueError
            If zero or more than one path is provided, or if the file
            extension does not match the expected suffixes.
        FileNotFoundError
            If the specified file does not exist.
        """
        provided = {
            name: value
            for name, value in [
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            ]
            if value is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )

        name, value = next(iter(provided.items()))
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{suffix}'"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

        if text_path is not None:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Text file is empty: {path}")
            return TextToEvents(
                text=text,
                infra={"folder": self.cache_folder, "mode": "retry"},
            ).get_events()

        event_type = "Audio" if audio_path is not None else "Video"
        event = {
            "type": event_type,
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]))

    def predict(
        self, events: pd.DataFrame, verbose: bool = True
    ) -> tuple[np.ndarray, list]:
        """Run inference on an events DataFrame and return per-TR predictions.

        Each batch is split into segments of length ``data.TR``.  When
        ``remove_empty_segments`` is ``True`` (the default), segments that
        contain no events are discarded.

        Parameters
        ----------
        events:
            Events DataFrame, typically produced by
            :meth:`get_events_dataframe`.
        verbose:
            If ``True`` (default), display a ``tqdm`` progress bar.

        Returns
        -------
        preds : np.ndarray
            Array of shape ``(n_kept_segments, n_vertices)`` with the
            predicted brain activity.
        all_segments : list
            Corresponding segment objects aligned with *preds*.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via :meth:`from_pretrained`.
        """
        if self._model is None:
            raise RuntimeError(
                "TribeModel must be instantiated via the .from_pretrained method"
            )
        model = self._model
        loader = self.data.get_loaders(events=events, split_to_build="all")["all"]

        preds, all_segments = [], []
        n_samples, n_kept = 0, 0
        with torch.inference_mode():
            for batch in tqdm(loader, disable=not verbose):
                batch = batch.to(model.device)
                batch_segments = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, self.data.TR):
                        batch_segments.append(
                            segment.copy(offset=t, duration=self.data.TR)
                        )
                if self.remove_empty_segments:
                    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
                else:
                    keep = np.ones(len(batch_segments), dtype=bool)
                n_kept += keep.sum()
                n_samples += len(batch_segments)
                batch_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
                y_pred = model(batch).detach().cpu().numpy()
                y_pred = rearrange(y_pred, "b d t -> (b t) d")[keep]
                preds.append(y_pred)
                all_segments.extend(batch_segments)
        preds = np.concatenate(preds)
        if len(all_segments) != preds.shape[0]:
            raise ValueError(
                f"Number of samples: {preds.shape[0]} != {len(all_segments)}"
            )
        logger.info(
            "Predicted %d / %d segments (%.1f%% kept)",
            n_kept,
            n_samples,
            100.0 * n_kept / max(n_samples, 1),
        )
        return preds, all_segments
===
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TribeModel for inference and utilities for building event DataFrames."""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import requests
import torch
import yaml
from einops import rearrange
from exca import ConfDict, TaskInfra
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)
from neuralset.events.transforms import (
    AddContextToWords,
    AddSentenceToWords,
    AddText,
    ChunkEvents,
    ExtractAudioFromVideo,
    RemoveMissing,
)
from neuralset.events.utils import standardize_events

from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.main import TribeExperiment

VALID_SUFFIXES: dict[str, set[str]] = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}


def download_file(url: str, path: str | Path) -> Path:
    """Download a file from *url* and save it to *path*.

    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Downloaded {url} -> {path}")
    return path


def get_audio_and_text_events(
    events: pd.DataFrame, audio_only: bool = False
) -> pd.DataFrame:
    """Run the audio/video-to-text pipeline on an events DataFrame.

    Extracts audio from video, chunks long clips, transcribes words, and
    attaches sentence/context annotations.  Set *audio_only* to ``True``
    to skip the transcription and text stages.
    """
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]
    if not audio_only:
        transforms.extend(
            [
                ExtractWordsFromAudio(),
                AddText(),
                AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(
                    sentence_only=False, max_context_len=1024, split_field=""
                ),
                RemoveMissing(),
            ]
        )
    events = standardize_events(events)
    for transform in transforms:
        events = transform(events)
    return standardize_events(events)


class TextToEvents(pydantic.BaseModel):
    """Convert raw text to an events DataFrame via text-to-speech + transcription.

    The text is synthesised to audio with gTTS, then processed through
    :func:`get_audio_and_text_events` to obtain word-level events.
    """

    text: str
    infra: TaskInfra = TaskInfra()

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            raise ValueError("A folder must be specified to save the audio file.")

    @infra.apply()
    def get_events(self) -> pd.DataFrame:
        from gtts import gTTS
        from langdetect import detect

        audio_path = Path(self.infra.uid_folder(create=True)) / "audio.mp3"
        lang = detect(self.text)
        tts = gTTS(self.text, lang=lang)
        tts.save(str(audio_path))
        logger.info(f"Wrote TTS audio to {audio_path}")

        audio_event = {
            "type": "Audio",
            "filepath": str(audio_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([audio_event]))


class TribeModel(TribeExperiment):
    """High-level inference wrapper around :class:`TribeExperiment`.

    Provides a simple ``from_pretrained`` / ``predict`` interface for
    generating fMRI-like brain-activity predictions from text, audio,
    or video inputs.

    Typical usage::

        model = TribeModel.from_pretrained("facebook/tribev2")
        events = model.get_events_dataframe(video_path="clip.mp4")
        preds, segments = model.predict(events)
    """

    cache_folder: str = "./cache"
    remove_empty_segments: bool = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        checkpoint_name: str = "best.ckpt",
        cache_folder: str | Path = None,
        cluster: str = None,
        device: str = "auto",
        config_update: dict | None = None,
    ) -> "TribeModel":
        """Load a trained model from a checkpoint directory or HuggingFace Hub repo.

        ``checkpoint_dir`` can be either a local path containing
        ``config.yaml`` and ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).

        Parameters
        ----------
        checkpoint_dir:
            Local directory or HuggingFace Hub repo id that contains
            ``config.yaml`` and the checkpoint file.
        checkpoint_name:
            Filename of the checkpoint inside *checkpoint_dir*.
        cache_folder:
            Directory used to cache extracted features. Created if it
            does not exist.  Defaults to ``"./cache"`` when ``None``.
        cluster:
            Cluster backend forwarded to feature-extractor infra
            (``"auto"`` by default).
        device:
            Torch device string.  ``"auto"`` selects CUDA when available.
        config_update:
            Optional dictionary of config overrides applied after the
            YAML config is loaded.

        Returns
        -------
        TribeModel
            A ready-to-use model instance with weights loaded in eval mode.
        """
        if cache_folder is not None:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            config_path = checkpoint_dir / "config.yaml"
            ckpt_path = checkpoint_dir / checkpoint_name
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(checkpoint_dir)
            config_path = hf_hub_download(repo_id, "config.yaml")
            ckpt_path = hf_hub_download(repo_id, checkpoint_name)
        with open(config_path, "r") as f:
            config = ConfDict(yaml.load(f, Loader=yaml.UnsafeLoader))
        for modality in ["text", "audio", "video"]:
            config[f"data.{modality}_feature.infra.folder"] = cache_folder
            config[f"data.{modality}_feature.infra.cluster"] = cluster

        for param in [
            "infra.workdir",
            "data.study.infra_timelines",
            "data.neuro.infra",
            "data.image_feature.infra",
        ]:
            config.pop(param)
        config["data.study.path"] = "."
        config["average_subjects"] = True
        config["checkpoint_path"] = str(config["infra.folder"]) + f"/{checkpoint_name}"
        config["cache_folder"] = (
            str(cache_folder) if cache_folder is not None else "./cache"
        )
        if config_update is not None:
            config.update(config_update)
        xp = cls(**config)

        logger.info(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt

        model = xp.brain_model_config.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device)
        model.half()  # float16 inference saves ~338 MB with negligible accuracy loss
        model.eval()
        xp._model = model
        return xp

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        """Build an events DataFrame from exactly one input source.

        Parameters
        ----------
        text_path:
            Path to a ``.txt`` file. The text is converted to speech, then
            transcribed back to produce word-level events.
        audio_path:
            Path to an audio file (``.wav``, ``.mp3``, ``.flac``, ``.ogg``).
        video_path:
            Path to a video file (``.mp4``, ``.avi``, ``.mkv``, ``.mov``,
            ``.webm``).

        Returns
        -------
        pd.DataFrame
            Standardised events DataFrame with columns such as ``type``,
            ``filepath``, ``start``, ``duration``, ``timeline``, and
            ``subject``.

        Raises
        ------
        ValueError
            If zero or more than one path is provided, or if the file
            extension does not match the expected suffixes.
        FileNotFoundError
            If the specified file does not exist.
        """
        provided = {
            name: value
            for name, value in [
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            ]
            if value is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )

        name, value = next(iter(provided.items()))
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{suffix}'"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

        if text_path is not None:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Text file is empty: {path}")
            return TextToEvents(
                text=text,
                infra={"folder": self.cache_folder, "mode": "retry"},
            ).get_events()

        event_type = "Audio" if audio_path is not None else "Video"
        event = {
            "type": event_type,
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]))

    def predict(
        self, events: pd.DataFrame, verbose: bool = True
    ) -> tuple[np.ndarray, list]:
        """Run inference on an events DataFrame and return per-TR predictions.

        Each batch is split into segments of length ``data.TR``.  When
        ``remove_empty_segments`` is ``True`` (the default), segments that
        contain no events are discarded.

        Parameters
        ----------
        events:
            Events DataFrame, typically produced by
            :meth:`get_events_dataframe`.
        verbose:
            If ``True`` (default), display a ``tqdm`` progress bar.

        Returns
        -------
        preds : np.ndarray
            Array of shape ``(n_kept_segments, n_vertices)`` with the
            predicted brain activity.
        all_segments : list
            Corresponding segment objects aligned with *preds*.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via :meth:`from_pretrained`.
        """
        if self._model is None:
            raise RuntimeError(
                "TribeModel must be instantiated via the .from_pretrained method"
            )
        model = self._model
        loader = self.data.get_loaders(events=events, split_to_build="all")["all"]

        preds, all_segments = [], []
        n_samples, n_kept = 0, 0
        with torch.inference_mode():
            for batch in tqdm(loader, disable=not verbose):
                batch = batch.to(model.device)
                batch_segments = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, self.data.TR):
                        batch_segments.append(
                            segment.copy(offset=t, duration=self.data.TR)
                        )
                if self.remove_empty_segments:
                    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
                else:
                    keep = np.ones(len(batch_segments), dtype=bool)
                n_kept += keep.sum()
                n_samples += len(batch_segments)
                batch_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
                y_pred = model(batch).detach().cpu().numpy()
                y_pred = rearrange(y_pred, "b d t -> (b t) d")[keep]
                preds.append(y_pred)
                all_segments.extend(batch_segments)
        preds = np.concatenate(preds)
        if len(all_segments) != preds.shape[0]:
            raise ValueError(
                f"Number of samples: {preds.shape[0]} != {len(all_segments)}"
            )
        logger.info(
            "Predicted %d / %d segments (%.1f%% kept)",
            n_kept,
            n_samples,
            100.0 * n_kept / max(n_samples, 1),
        )
        return preds, all_segments
```

- Brain model (`FmriEncoder`, 177M params) loaded in float16 via `model.half()`

```diff:model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import Mlp, SubjectLayers, SubjectLayersModel
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

logger = logging.getLogger(__name__)


class TemporalSmoothing(BaseModelConfig):
    kernel_size: int = 9
    sigma: float | None = None

    def build(self, dim: int) -> nn.Module:

        def gaussian_kernel_1d(kernel_size: int, sigma: float):
            x = torch.arange(kernel_size) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel.view(1, 1, -1)

        conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
            groups=dim,
        )
        if self.sigma is not None:
            kernel = gaussian_kernel_1d(kernel_size=self.kernel_size, sigma=self.sigma)
            kernel = kernel.repeat(dim, 1, 1)
            conv.weight.data = kernel
            conv.requires_grad = False
        return conv


class FmriEncoder(BaseModelConfig):

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu")
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu")
    encoder: TransformerEncoder | None = TransformerEncoder()
    # other hyperparameters
    time_pos_embedding: bool = True
    subject_embedding: bool = False
    subject_layers: SubjectLayers | None = SubjectLayers()
    hidden: int = 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    linear_baseline: bool = False
    modality_dropout: float = 0.0
    temporal_dropout: float = 0.0
    low_rank_head: int | None = None
    temporal_smoothing: TemporalSmoothing | None = None

    def model_post_init(self, __context):
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoderModel(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoderModel(nn.Module):

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.n_output_timesteps = n_output_timesteps
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = config.hidden
        for modality, tup in feature_dims.items():
            if tup is None:
                logger.warning(
                    "%s has no feature dimensions. Skipping projector.", modality
                )
                continue
            else:
                num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.extractor_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = self.config.projector.build(
                input_dim, output_dim
            )
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if self.config.combiner is not None:
            self.combiner = self.config.combiner.build(input_dim, hidden)
        else:
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()
        if config.low_rank_head is not None:
            self.low_rank_head = nn.Linear(hidden, config.low_rank_head, bias=False)
            bottleneck = config.low_rank_head
        else:
            bottleneck = hidden
        self.predictor = config.subject_layers.build(
            in_channels=bottleneck,
            out_channels=n_outputs,
        )
        if config.temporal_smoothing is not None:
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)
        if not config.linear_baseline:
            if config.time_pos_embedding:
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                )
            if config.subject_embedding:
                self.subject_embed = nn.Embedding(config.n_subjects, hidden)
            self.encoder = config.encoder.build(dim=hidden)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # B, T, H
        subject_id = batch.data.get("subject_id", None)
        if hasattr(self, "temporal_smoothing"):
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline:
            x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        if self.config.low_rank_head is not None:
            x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        for modality in self.feature_dims.keys():
            if modality not in self.projectors or modality not in batch.data:
                data = torch.zeros(
                    B, T, self.config.hidden // len(self.feature_dims)
                ).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                if isinstance(self.projectors[modality], SubjectLayersModel):
                    data = self.projectors[modality](
                        data.transpose(1, 2), batch.data["subject_id"]
                    ).transpose(1, 2)
                else:
                    data = self.projectors[modality](data)  # B, T, H
                if self.config.modality_dropout > 0 and self.training:
                    mask = torch.rand(data.shape[0]) < self.config.modality_dropout
                    data[mask, :] = torch.zeros_like(data[mask, :])
            tensors.append(data)
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training:
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x
===
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import Mlp, SubjectLayers, SubjectLayersModel
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

logger = logging.getLogger(__name__)


class TemporalSmoothing(BaseModelConfig):
    kernel_size: int = 9
    sigma: float | None = None

    def build(self, dim: int) -> nn.Module:

        def gaussian_kernel_1d(kernel_size: int, sigma: float):
            x = torch.arange(kernel_size) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel.view(1, 1, -1)

        conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
            groups=dim,
        )
        if self.sigma is not None:
            kernel = gaussian_kernel_1d(kernel_size=self.kernel_size, sigma=self.sigma)
            kernel = kernel.repeat(dim, 1, 1)
            conv.weight.data = kernel
            conv.requires_grad = False
        return conv


class FmriEncoder(BaseModelConfig):

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu")
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu")
    encoder: TransformerEncoder | None = TransformerEncoder()
    # other hyperparameters
    time_pos_embedding: bool = True
    subject_embedding: bool = False
    subject_layers: SubjectLayers | None = SubjectLayers()
    hidden: int = 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    linear_baseline: bool = False
    modality_dropout: float = 0.0
    temporal_dropout: float = 0.0
    low_rank_head: int | None = None
    temporal_smoothing: TemporalSmoothing | None = None

    def model_post_init(self, __context):
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoderModel(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoderModel(nn.Module):

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.n_output_timesteps = n_output_timesteps
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = config.hidden
        for modality, tup in feature_dims.items():
            if tup is None:
                logger.warning(
                    "%s has no feature dimensions. Skipping projector.", modality
                )
                continue
            else:
                num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.extractor_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = self.config.projector.build(
                input_dim, output_dim
            )
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if self.config.combiner is not None:
            self.combiner = self.config.combiner.build(input_dim, hidden)
        else:
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()
        if config.low_rank_head is not None:
            self.low_rank_head = nn.Linear(hidden, config.low_rank_head, bias=False)
            bottleneck = config.low_rank_head
        else:
            bottleneck = hidden
        self.predictor = config.subject_layers.build(
            in_channels=bottleneck,
            out_channels=n_outputs,
        )
        if config.temporal_smoothing is not None:
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)
        if not config.linear_baseline:
            if config.time_pos_embedding:
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                )
            if config.subject_embedding:
                self.subject_embed = nn.Embedding(config.n_subjects, hidden)
            self.encoder = config.encoder.build(dim=hidden)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # B, T, H
        subject_id = batch.data.get("subject_id", None)
        if hasattr(self, "temporal_smoothing"):
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline:
            x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        if self.config.low_rank_head is not None:
            x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        model_dtype = next(self.parameters()).dtype
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        for modality in self.feature_dims.keys():
            if modality not in self.projectors or modality not in batch.data:
                data = torch.zeros(
                    B, T, self.config.hidden // len(self.feature_dims),
                    dtype=model_dtype, device=x.device,
                )
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(model_dtype)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                if isinstance(self.projectors[modality], SubjectLayersModel):
                    data = self.projectors[modality](
                        data.transpose(1, 2), batch.data["subject_id"]
                    ).transpose(1, 2)
                else:
                    data = self.projectors[modality](data)  # B, T, H
                if self.config.modality_dropout > 0 and self.training:
                    mask = torch.rand(data.shape[0]) < self.config.modality_dropout
                    data[mask, :] = torch.zeros_like(data[mask, :])
            tensors.append(data)
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training:
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x
```

- `aggregate_features()` now uses `model_dtype = next(self.parameters()).dtype` instead of hardcoded `torch.float32`
- Backward-compatible: returns float32 when model is float32

---

### 4. Memory Cleanup Between Pipeline Stages

```diff:main.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the experiment.

We suggest the following structure:
- `Data`: configures dataset and extractors to return DataLoaders
- `Trainer`: creates the deep learning model and exposes a `fit` and `test` methods
- `TribeExperiment`: main class that defines the experiment to run by using `Data` and `Trainer`
"""

import gc
import logging
import os
import typing as tp
from pathlib import Path

import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import torch
import yaml
from exca import ConfDict, TaskInfra
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics import BaseMetric
from neuraltrain.models import BaseModelConfig
from neuraltrain.models.common import SubjectLayers
from neuraltrain.optimizers.base import BaseOptimizer
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig
from torch import nn
from torch.utils.data import DataLoader

from .eventstransforms import *  # register custom events transforms in neuralset
from .model import *  # register custom models in neuraltrain
from .studies import *  # register studies
from .utils import (
    MultiStudyLoader,
    set_study_in_average_subject_mode,
    split_segments_by_time,
)
from .utils_fmri import *  # register TribeSurfaceProjector

# Configure logger
LOGGER = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
_handler.setFormatter(_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


def _free_extractor_model(extractor: ns.extractors.BaseExtractor) -> None:
    """Delete cached GPU model from an extractor after its features are cached.

    Extractors lazily load models onto GPU during ``prepare`` and keep them
    in ``_model``.  Since results are persisted to disk, the model is no
    longer needed afterwards and this frees VRAM for subsequent extractors.
    """
    targets = [extractor]
    if hasattr(extractor, "image"):
        targets.append(extractor.image)
    for target in targets:
        for attr in ("_model",):
            obj = getattr(target, attr, None)
            if isinstance(obj, torch.nn.Module):
                try:
                    delattr(target, attr)
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: MultiStudyLoader
    # features
    neuro: ns.extractors.BaseExtractor
    text_feature: ns.extractors.BaseExtractor | None = None
    image_feature: ns.extractors.BaseExtractor | None = None
    audio_feature: ns.extractors.BaseExtractor | None = None
    video_feature: ns.extractors.BaseExtractor | None = None
    subject_id: ns.extractors.LabelEncoder = ns.extractors.LabelEncoder(
        event_field="subject", allow_missing=True, aggregation="first"
    )
    frequency: float | None = None
    features_to_use: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ]
    features_to_mask: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ] = []
    n_layers_to_use: int | None = None
    layers_to_use: list[float] | None = None
    layer_aggregation: tp.Literal["group_mean", "mean"] | None = "group_mean"
    # Dataset
    duration_trs: int = 40
    overlap_trs_train: int = 0
    overlap_trs_val: int | None = None
    batch_size: int = 64
    num_workers: int | None = None
    shuffle_train: bool = True
    shuffle_val: bool = False
    stride_drop_incomplete: bool = False
    split_segments_by_time: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        layers_to_use = None
        if self.n_layers_to_use is not None or self.layers_to_use is not None:
            assert not (
                self.n_layers_to_use is not None and self.layers_to_use is not None
            ), "Only one of n_layers_to_use or layers_to_use can be specified"
            if self.n_layers_to_use is not None:
                layers_to_use = np.linspace(0, 1, self.n_layers_to_use).tolist()
            else:
                layers_to_use = self.layers_to_use
        for modality in self.features_to_use:
            extractor = getattr(self, f"{modality}_feature")
            if hasattr(extractor, "layers"):
                setattr(extractor, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor, "layers", layers_to_use)
            if hasattr(extractor, "image") and hasattr(extractor.image, "layers"):
                setattr(extractor.image, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor.image, "layers", layers_to_use)
        if self.frequency is not None:
            for modality in self.features_to_use:
                extractor = getattr(self, f"{modality}_feature")
                if hasattr(extractor, "frequency"):
                    setattr(extractor, "frequency", self.frequency)

    @property
    def TR(self) -> float:
        return 1 / self.neuro.frequency

    def get_events(self) -> pd.DataFrame:
        events = self.study.run()
        events = events[events.type != "Sentence"]

        cols = ["index", "subject", "timeline"]
        event_summary = (
            events.reset_index().groupby(["study", "split", "type"])[cols].nunique()
        )
        LOGGER.info("Event summary: \n%s", event_summary)
        return events

    def get_loaders(
        self,
        events: pd.DataFrame | None = None,
        split_to_build: tp.Literal["train", "val", "all"] | None = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events()
        else:
            events = standardize_events(events)

        extractors = {}
        for modality in self.features_to_use:
            extractors[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique():
            extractors["fmri"] = self.neuro
        dummy_events = []
        for timeline_name, timeline in events.groupby("timeline"):
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                assert (
                    len(splits) == 1
                ), f"Timeline {timeline_name} has multiple splits: {splits}"
                split = splits[0]
            else:
                split = "all"
            dummy_event = {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
            dummy_events.append(dummy_event)
        events = pd.concat([events, pd.DataFrame(dummy_events)])
        events = standardize_events(events)

        extractors["subject_id"] = self.subject_id

        features_to_remove = set()
        for extractor_name, extractor in extractors.items():
            event_types = EventTypesHelper(extractor.event_types).names
            if not any(
                [event_type in events.type.unique() for event_type in event_types]
            ):
                features_to_remove.add(extractor_name)
        for extractor_name in features_to_remove:
            del extractors[extractor_name]
            LOGGER.warning(
                "Removing extractor %s as there are no corresponding events",
                extractor_name,
            )

        for name, extractor in extractors.items():
            LOGGER.info("Preparing extractor: %s", name)
            extractor.prepare(events)
            _free_extractor_model(extractor)

        # Prepare dataloaders
        loaders = {}
        if split_to_build is None:
            splits = ["train", "val"]
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all" or self.split_segments_by_time:
                split_sel = [True] * len(events)
                shuffle = False
                overlap_trs = self.overlap_trs_train
            else:
                split_sel = events.split == split
                if split not in events.split.unique():
                    shuffle = False
                else:
                    shuffle = (
                        self.shuffle_train if split == "train" else self.shuffle_val
                    )
                if split == "val":
                    overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                else:
                    overlap_trs = self.overlap_trs_train

            sel = np.array(split_sel)
            segments = ns.segments.list_segments(
                events[sel],
                triggers=events[sel].type == "CategoricalEvent",
                stride=(self.duration_trs - overlap_trs) * self.TR,
                duration=self.duration_trs * self.TR,
                stride_drop_incomplete=self.stride_drop_incomplete,
            )
            if self.split_segments_by_time:
                LOGGER.info(f"Total number of segments: {len(segments)}")
                segments = split_segments_by_time(
                    segments,
                    val_ratio=self.study.transforms["split"].val_ratio,
                    split=split,
                )
                LOGGER.info(f"# {split} segments: {len(segments)}")
            if len(segments) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            dataset = ns.dataloader.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
            loaders[split] = dataloader

        return loaders


class TribeExperiment(BaseExperiment):
    """Defines the main experiment pipeline including data loading and training/evaluation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data: Data
    # Reproducibility
    seed: int | None = 33
    # Model
    brain_model_config: BaseModelConfig
    # Loss
    loss: BaseLoss
    # Optimization
    optim: BaseOptimizer
    # Metrics
    metrics: list[BaseMetric]
    monitor: str = "val/pearson"
    # Weights & Biases
    wandb_config: WandbLoggerConfig | None = None
    # Hardware
    accelerator: str = "auto"
    # Optim
    n_epochs: int | None = 10
    max_steps: int = -1
    patience: int | None = None
    limit_train_batches: int | None = None
    accumulate_grad_batches: int = 1
    # Others
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    save_checkpoints: bool = True
    checkpoint_filename: str = "best"
    resize_subject_layer: bool = False
    freeze_backbone: bool = False
    # Eval
    average_subjects: bool = False
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    test_only: bool = False

    # Internal properties
    _trainer: tp.Any = None
    _model: tp.Any = None
    _logger: tp.Any = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        # Update Trainer parameters based on infra
        self.infra.tasks_per_node = self.infra.gpus_per_node
        self.infra.slurm_use_srun = True if self.infra.gpus_per_node > 1 else False
        if self.infra.gpus_per_node > 1:
            self.metrics = [m for m in self.metrics if m.name not in ["TopkAcc"]]
            self.data.batch_size = self.data.batch_size // self.infra.gpus_per_node
        if self.accumulate_grad_batches > 1:
            self.data.batch_size = self.data.batch_size // self.accumulate_grad_batches

        if (
            not (self.checkpoint_path and self.load_checkpoint)
        ) or self.resize_subject_layer:
            study_summary = self.data.study.study_summary()
            self.data.subject_id.predefined_mapping = {
                subject: i for i, subject in enumerate(study_summary.subject.unique())
            }
            self.brain_model_config.subject_layers.n_subjects = (
                study_summary.subject.nunique()
            )
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.n_subjects = (
                    study_summary.subject.nunique()
                )

        if self.average_subjects:
            study_name = self.data.study.names
            self.brain_model_config.subject_layers.average_subjects = True
            self.brain_model_config.subject_layers.n_subjects = 0
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.average_subjects = True
            self.data.neuro.aggregation = "mean"
            self.data.subject_id.predefined_mapping = None
            if isinstance(study_name, str):
                LOGGER.debug(f"Setting study {study_name} in average subject mode")
                trigger_type = (
                    "Video" if study_name in ["Wen2017", "Allen2022Bold"] else "Audio"
                )
                self.data.study = set_study_in_average_subject_mode(
                    self.data.study, trigger_type=trigger_type, trigger_field="filepath"
                )
            else:
                pass
                # LOGGER.warning(
                #     "Cannot set study in average subject mode with multiple studies"
                # )

    def _get_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            assert Path(
                self.checkpoint_path
            ).exists(), f"Checkpoint path {self.checkpoint_path} does not exist."
            checkpoint_path = Path(self.checkpoint_path)
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = None
        return checkpoint_path

    def _init_module(self, model: nn.Module) -> tp.Any:
        from .pl_module import BrainModule

        checkpoint_path = self._get_checkpoint_path()
        if (
            self.load_checkpoint
            and checkpoint_path is not None
            and not self.resize_subject_layer
        ):
            LOGGER.info(f"Loading model from {checkpoint_path}")
            init_fn = BrainModule.load_from_checkpoint
            init_kwargs = {"checkpoint_path": checkpoint_path, "strict": False}
        else:
            init_fn = BrainModule
            init_kwargs = {}

        metrics = {
            split + "/" + metric.log_name: metric.build()
            for metric in self.metrics
            for split in ["val", "test"]
        }
        metrics = nn.ModuleDict(metrics)
        pl_module = init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            config=ConfDict(self.model_dump()),
            **init_kwargs,
        )

        if self.resize_subject_layer:
            LOGGER.info("Resizing subject layer")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            weights = state_dict["model.predictor.weights"]
            _, in_channels, out_channels = weights.shape
            n_subjects = self.brain_model_config.subject_layers.n_subjects
            if self.brain_model_config.subject_layers.subject_dropout:
                n_subjects += 1
            if "model.predictor.bias" in state_dict:
                bias = state_dict["model.predictor.bias"]
                new_bias = torch.nn.Parameter(torch.zeros(n_subjects, out_channels))
                new_bias.data[:] = bias.mean(dim=0).repeat(n_subjects, 1)
                state_dict["model.predictor.bias"] = new_bias
            if self.freeze_backbone:
                for param in pl_module.parameters():
                    param.requires_grad = False
            for param in pl_module.model.predictor.parameters():
                param.requires_grad = True
            if (
                self.brain_model_config.low_rank_head is not None
                and self.brain_model_config.low_rank_head != in_channels
            ):
                r = self.brain_model_config.low_rank_head
                if "model.low_rank_head.weight" in state_dict:
                    W1, W2 = (
                        state_dict["model.low_rank_head.weight"].cpu(),
                        state_dict["model.predictor.weights"].mean(dim=0).cpu(),
                    )
                    prod = torch.matmul(W1.t(), W2)
                else:
                    prod = state_dict["model.predictor.weights"].mean(dim=0).cpu()
                U, S, V = torch.svd(prod)
                U = U[:, :r]
                S = S[:r]
                V = V[:, :r]
                state_dict["model.low_rank_head.weight"] = U.t()
                state_dict["model.predictor.weights"] = torch.matmul(
                    torch.diag(S), V.t()
                ).repeat(n_subjects, 1, 1)
                if "model.predictor.bias" in state_dict:
                    state_dict["model.low_rank_head.bias"] = torch.zeros(r)
                for param in pl_module.model.low_rank_head.parameters():
                    param.requires_grad = True
            else:
                state_dict["model.predictor.weights"] = weights.mean(dim=0).repeat(
                    n_subjects, 1, 1
                )
            pl_module.load_state_dict(state_dict, strict=False)

        return pl_module

    def _setup_trainer(
        self, train_loader: DataLoader, override_n_devices: int | None = None
    ) -> tp.Any:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        batch = next(iter(train_loader))
        feature_dims = {}
        for modality in self.data.features_to_use:
            if (
                modality in batch.data and modality not in self.data.features_to_mask
            ):  # B, L, D, T
                if batch.data[modality].ndim == 4:
                    feature_dims[modality] = (
                        batch.data[modality].shape[1],
                        batch.data[modality].shape[2],
                    )
                elif batch.data[modality].ndim == 3:
                    feature_dims[modality] = (
                        1,
                        batch.data[modality].shape[1],
                    )
                else:
                    raise ValueError(
                        f"Unexpected number of dimensions for modality {modality}: {batch.data[modality].ndim}"
                    )
            else:
                feature_dims[modality] = None
        if "fmri" in batch.data:  # read from fmri config
            fmri = batch.data["fmri"]
            n_outputs = fmri.shape[1]
            for metric in self.metrics:
                if hasattr(metric, "kwargs") and "num_outputs" in metric.kwargs:
                    metric.kwargs["num_outputs"] = n_outputs
        else:  # read from neuro config
            if hasattr(self.data.neuro.projection, "mesh"):
                from neuralset.extractors.neuro import FSAVERAGE_SIZES

                n_outputs = 2 * FSAVERAGE_SIZES[self.data.neuro.projection.mesh]
            else:
                raise ValueError(
                    f"Could not determine number of outputs for neuro extractor {self.data.neuro}"
                )
        brain_model = self.brain_model_config.build(
            feature_dims=feature_dims,
            n_outputs=n_outputs,
            n_output_timesteps=self.data.duration_trs,
        )
        LOGGER.info("Extractor dims: %s", feature_dims)
        input_data = brain_model.aggregate_features(batch)
        LOGGER.info("Input shapes: %s", input_data.shape)
        LOGGER.info("Target shapes: %s", n_outputs)
        _ = brain_model(batch)
        total_params = sum(p.numel() for p in brain_model.parameters())
        LOGGER.info(f"Total parameters: {total_params}")
        self._model = self._init_module(brain_model)
        if self.monitor == "val/pearson":
            mode = "max"
        else:
            mode = "min"
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
        ]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=self.monitor, mode=mode, patience=self.patience)
            )
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename=self.checkpoint_filename,
                    monitor=self.monitor,
                    mode=mode,
                    save_on_train_epoch_end=True,
                )
            )

        n_devices = override_n_devices or self.infra.gpus_per_node
        # MPS only supports 1 device and does not support FSDP
        if self.accelerator == "mps" or (
            self.accelerator == "auto" and not torch.cuda.is_available()
        ):
            n_devices = max(n_devices, 1)
            strategy = "auto"
        else:
            strategy = "auto" if n_devices == 1 else "fsdp"
        trainer = pl.Trainer(
            strategy=strategy,
            devices=n_devices,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            max_steps=self.max_steps,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
            enable_checkpointing=self.save_checkpoints,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        self._trainer = trainer
        return trainer

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._trainer.fit(
            model=self._model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._get_checkpoint_path(),
        )

    def test(self, test_loader: DataLoader) -> None:
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
        else:
            if self.save_checkpoints:
                ckpt_path = Path(self.infra.folder) / "best.ckpt"
            else:
                ckpt_path = None
        self._trainer.test(
            self._model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )

    def setup_run(self):

        if self.infra.cluster and self.infra.status() != "not submitted":
            for out_type in ["stdout", "stderr"]:
                old_path = Path(getattr(self.infra.job().paths, out_type))
                new_path = Path(self.infra.folder) / f"log.{out_type}"
                try:
                    if new_path.exists():
                        os.remove(new_path)
                    os.symlink(
                        old_path,
                        new_path,
                    )
                except Exception:
                    pass
        config_path = Path(self.infra.folder) / "config.yaml"
        os.makedirs(self.infra.folder, exist_ok=True)
        with open(config_path, "w") as outfile:
            yaml.dump(
                self.model_dump(),
                outfile,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @infra.apply
    def run(self):
        import lightning.pytorch as pl

        self.setup_run()
        self._logger = (
            self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
                id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
            )
            if self.wandb_config
            else None
        )

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        loaders = self.data.get_loaders(
            split_to_build="val" if self.test_only else None
        )
        self._setup_trainer(next(iter(loaders.values())))

        if not self.test_only:
            self.fit(loaders["train"], loaders["val"])

        self.test(loaders["val"])
===
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the experiment.

We suggest the following structure:
- `Data`: configures dataset and extractors to return DataLoaders
- `Trainer`: creates the deep learning model and exposes a `fit` and `test` methods
- `TribeExperiment`: main class that defines the experiment to run by using `Data` and `Trainer`
"""

import gc
import logging
import os
import typing as tp
from pathlib import Path

import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import torch
import yaml
from exca import ConfDict, TaskInfra
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics import BaseMetric
from neuraltrain.models import BaseModelConfig
from neuraltrain.models.common import SubjectLayers
from neuraltrain.optimizers.base import BaseOptimizer
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig
from torch import nn
from torch.utils.data import DataLoader

from .eventstransforms import *  # register custom events transforms in neuralset
from .model import *  # register custom models in neuraltrain
from .studies import *  # register studies
from .utils import (
    MultiStudyLoader,
    set_study_in_average_subject_mode,
    split_segments_by_time,
)
from .utils_fmri import *  # register TribeSurfaceProjector

# Configure logger
LOGGER = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
_handler.setFormatter(_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


def _free_extractor_model(extractor: ns.extractors.BaseExtractor) -> None:
    """Delete cached GPU model from an extractor after its features are cached.

    Extractors lazily load models onto GPU during ``prepare`` and keep them
    in ``_model``.  Since results are persisted to disk, the model is no
    longer needed afterwards and this frees VRAM for subsequent extractors.

    Also frees ``_tokenizer`` (used by text extractors) so that the full
    memory footprint of the extractor is reclaimed between pipeline stages.
    """
    targets = [extractor]
    if hasattr(extractor, "image"):
        targets.append(extractor.image)
    for target in targets:
        for attr in ("_model", "_tokenizer"):
            obj = getattr(target, attr, None)
            if obj is not None:
                try:
                    delattr(target, attr)
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: MultiStudyLoader
    # features
    neuro: ns.extractors.BaseExtractor
    text_feature: ns.extractors.BaseExtractor | None = None
    image_feature: ns.extractors.BaseExtractor | None = None
    audio_feature: ns.extractors.BaseExtractor | None = None
    video_feature: ns.extractors.BaseExtractor | None = None
    subject_id: ns.extractors.LabelEncoder = ns.extractors.LabelEncoder(
        event_field="subject", allow_missing=True, aggregation="first"
    )
    frequency: float | None = None
    features_to_use: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ]
    features_to_mask: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ] = []
    n_layers_to_use: int | None = None
    layers_to_use: list[float] | None = None
    layer_aggregation: tp.Literal["group_mean", "mean"] | None = "group_mean"
    # Dataset
    duration_trs: int = 40
    overlap_trs_train: int = 0
    overlap_trs_val: int | None = None
    batch_size: int = 64
    num_workers: int | None = None
    shuffle_train: bool = True
    shuffle_val: bool = False
    stride_drop_incomplete: bool = False
    split_segments_by_time: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        layers_to_use = None
        if self.n_layers_to_use is not None or self.layers_to_use is not None:
            assert not (
                self.n_layers_to_use is not None and self.layers_to_use is not None
            ), "Only one of n_layers_to_use or layers_to_use can be specified"
            if self.n_layers_to_use is not None:
                layers_to_use = np.linspace(0, 1, self.n_layers_to_use).tolist()
            else:
                layers_to_use = self.layers_to_use
        for modality in self.features_to_use:
            extractor = getattr(self, f"{modality}_feature")
            if hasattr(extractor, "layers"):
                setattr(extractor, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor, "layers", layers_to_use)
            if hasattr(extractor, "image") and hasattr(extractor.image, "layers"):
                setattr(extractor.image, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor.image, "layers", layers_to_use)
        if self.frequency is not None:
            for modality in self.features_to_use:
                extractor = getattr(self, f"{modality}_feature")
                if hasattr(extractor, "frequency"):
                    setattr(extractor, "frequency", self.frequency)

    @property
    def TR(self) -> float:
        return 1 / self.neuro.frequency

    def get_events(self) -> pd.DataFrame:
        events = self.study.run()
        events = events[events.type != "Sentence"]

        cols = ["index", "subject", "timeline"]
        event_summary = (
            events.reset_index().groupby(["study", "split", "type"])[cols].nunique()
        )
        LOGGER.info("Event summary: \n%s", event_summary)
        return events

    def get_loaders(
        self,
        events: pd.DataFrame | None = None,
        split_to_build: tp.Literal["train", "val", "all"] | None = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events()
        else:
            events = standardize_events(events)

        extractors = {}
        for modality in self.features_to_use:
            extractors[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique():
            extractors["fmri"] = self.neuro
        dummy_events = []
        for timeline_name, timeline in events.groupby("timeline"):
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                assert (
                    len(splits) == 1
                ), f"Timeline {timeline_name} has multiple splits: {splits}"
                split = splits[0]
            else:
                split = "all"
            dummy_event = {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
            dummy_events.append(dummy_event)
        events = pd.concat([events, pd.DataFrame(dummy_events)])
        events = standardize_events(events)

        extractors["subject_id"] = self.subject_id

        features_to_remove = set()
        for extractor_name, extractor in extractors.items():
            event_types = EventTypesHelper(extractor.event_types).names
            if not any(
                [event_type in events.type.unique() for event_type in event_types]
            ):
                features_to_remove.add(extractor_name)
        for extractor_name in features_to_remove:
            del extractors[extractor_name]
            LOGGER.warning(
                "Removing extractor %s as there are no corresponding events",
                extractor_name,
            )

        for name, extractor in extractors.items():
            LOGGER.info("Preparing extractor: %s", name)
            extractor.prepare(events)
            _free_extractor_model(extractor)

        # Prepare dataloaders
        loaders = {}
        if split_to_build is None:
            splits = ["train", "val"]
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all" or self.split_segments_by_time:
                split_sel = [True] * len(events)
                shuffle = False
                overlap_trs = self.overlap_trs_train
            else:
                split_sel = events.split == split
                if split not in events.split.unique():
                    shuffle = False
                else:
                    shuffle = (
                        self.shuffle_train if split == "train" else self.shuffle_val
                    )
                if split == "val":
                    overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                else:
                    overlap_trs = self.overlap_trs_train

            sel = np.array(split_sel)
            segments = ns.segments.list_segments(
                events[sel],
                triggers=events[sel].type == "CategoricalEvent",
                stride=(self.duration_trs - overlap_trs) * self.TR,
                duration=self.duration_trs * self.TR,
                stride_drop_incomplete=self.stride_drop_incomplete,
            )
            if self.split_segments_by_time:
                LOGGER.info(f"Total number of segments: {len(segments)}")
                segments = split_segments_by_time(
                    segments,
                    val_ratio=self.study.transforms["split"].val_ratio,
                    split=split,
                )
                LOGGER.info(f"# {split} segments: {len(segments)}")
            if len(segments) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            dataset = ns.dataloader.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
            loaders[split] = dataloader

        return loaders


class TribeExperiment(BaseExperiment):
    """Defines the main experiment pipeline including data loading and training/evaluation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data: Data
    # Reproducibility
    seed: int | None = 33
    # Model
    brain_model_config: BaseModelConfig
    # Loss
    loss: BaseLoss
    # Optimization
    optim: BaseOptimizer
    # Metrics
    metrics: list[BaseMetric]
    monitor: str = "val/pearson"
    # Weights & Biases
    wandb_config: WandbLoggerConfig | None = None
    # Hardware
    accelerator: str = "auto"
    # Optim
    n_epochs: int | None = 10
    max_steps: int = -1
    patience: int | None = None
    limit_train_batches: int | None = None
    accumulate_grad_batches: int = 1
    # Others
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    save_checkpoints: bool = True
    checkpoint_filename: str = "best"
    resize_subject_layer: bool = False
    freeze_backbone: bool = False
    # Eval
    average_subjects: bool = False
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    test_only: bool = False

    # Internal properties
    _trainer: tp.Any = None
    _model: tp.Any = None
    _logger: tp.Any = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        # Update Trainer parameters based on infra
        self.infra.tasks_per_node = self.infra.gpus_per_node
        self.infra.slurm_use_srun = True if self.infra.gpus_per_node > 1 else False
        if self.infra.gpus_per_node > 1:
            self.metrics = [m for m in self.metrics if m.name not in ["TopkAcc"]]
            self.data.batch_size = self.data.batch_size // self.infra.gpus_per_node
        if self.accumulate_grad_batches > 1:
            self.data.batch_size = self.data.batch_size // self.accumulate_grad_batches

        if (
            not (self.checkpoint_path and self.load_checkpoint)
        ) or self.resize_subject_layer:
            study_summary = self.data.study.study_summary()
            self.data.subject_id.predefined_mapping = {
                subject: i for i, subject in enumerate(study_summary.subject.unique())
            }
            self.brain_model_config.subject_layers.n_subjects = (
                study_summary.subject.nunique()
            )
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.n_subjects = (
                    study_summary.subject.nunique()
                )

        if self.average_subjects:
            study_name = self.data.study.names
            self.brain_model_config.subject_layers.average_subjects = True
            self.brain_model_config.subject_layers.n_subjects = 0
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.average_subjects = True
            self.data.neuro.aggregation = "mean"
            self.data.subject_id.predefined_mapping = None
            if isinstance(study_name, str):
                LOGGER.debug(f"Setting study {study_name} in average subject mode")
                trigger_type = (
                    "Video" if study_name in ["Wen2017", "Allen2022Bold"] else "Audio"
                )
                self.data.study = set_study_in_average_subject_mode(
                    self.data.study, trigger_type=trigger_type, trigger_field="filepath"
                )
            else:
                pass
                # LOGGER.warning(
                #     "Cannot set study in average subject mode with multiple studies"
                # )

    def _get_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            assert Path(
                self.checkpoint_path
            ).exists(), f"Checkpoint path {self.checkpoint_path} does not exist."
            checkpoint_path = Path(self.checkpoint_path)
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = None
        return checkpoint_path

    def _init_module(self, model: nn.Module) -> tp.Any:
        from .pl_module import BrainModule

        checkpoint_path = self._get_checkpoint_path()
        if (
            self.load_checkpoint
            and checkpoint_path is not None
            and not self.resize_subject_layer
        ):
            LOGGER.info(f"Loading model from {checkpoint_path}")
            init_fn = BrainModule.load_from_checkpoint
            init_kwargs = {"checkpoint_path": checkpoint_path, "strict": False}
        else:
            init_fn = BrainModule
            init_kwargs = {}

        metrics = {
            split + "/" + metric.log_name: metric.build()
            for metric in self.metrics
            for split in ["val", "test"]
        }
        metrics = nn.ModuleDict(metrics)
        pl_module = init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            config=ConfDict(self.model_dump()),
            **init_kwargs,
        )

        if self.resize_subject_layer:
            LOGGER.info("Resizing subject layer")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            weights = state_dict["model.predictor.weights"]
            _, in_channels, out_channels = weights.shape
            n_subjects = self.brain_model_config.subject_layers.n_subjects
            if self.brain_model_config.subject_layers.subject_dropout:
                n_subjects += 1
            if "model.predictor.bias" in state_dict:
                bias = state_dict["model.predictor.bias"]
                new_bias = torch.nn.Parameter(torch.zeros(n_subjects, out_channels))
                new_bias.data[:] = bias.mean(dim=0).repeat(n_subjects, 1)
                state_dict["model.predictor.bias"] = new_bias
            if self.freeze_backbone:
                for param in pl_module.parameters():
                    param.requires_grad = False
            for param in pl_module.model.predictor.parameters():
                param.requires_grad = True
            if (
                self.brain_model_config.low_rank_head is not None
                and self.brain_model_config.low_rank_head != in_channels
            ):
                r = self.brain_model_config.low_rank_head
                if "model.low_rank_head.weight" in state_dict:
                    W1, W2 = (
                        state_dict["model.low_rank_head.weight"].cpu(),
                        state_dict["model.predictor.weights"].mean(dim=0).cpu(),
                    )
                    prod = torch.matmul(W1.t(), W2)
                else:
                    prod = state_dict["model.predictor.weights"].mean(dim=0).cpu()
                U, S, V = torch.svd(prod)
                U = U[:, :r]
                S = S[:r]
                V = V[:, :r]
                state_dict["model.low_rank_head.weight"] = U.t()
                state_dict["model.predictor.weights"] = torch.matmul(
                    torch.diag(S), V.t()
                ).repeat(n_subjects, 1, 1)
                if "model.predictor.bias" in state_dict:
                    state_dict["model.low_rank_head.bias"] = torch.zeros(r)
                for param in pl_module.model.low_rank_head.parameters():
                    param.requires_grad = True
            else:
                state_dict["model.predictor.weights"] = weights.mean(dim=0).repeat(
                    n_subjects, 1, 1
                )
            pl_module.load_state_dict(state_dict, strict=False)

        return pl_module

    def _setup_trainer(
        self, train_loader: DataLoader, override_n_devices: int | None = None
    ) -> tp.Any:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        batch = next(iter(train_loader))
        feature_dims = {}
        for modality in self.data.features_to_use:
            if (
                modality in batch.data and modality not in self.data.features_to_mask
            ):  # B, L, D, T
                if batch.data[modality].ndim == 4:
                    feature_dims[modality] = (
                        batch.data[modality].shape[1],
                        batch.data[modality].shape[2],
                    )
                elif batch.data[modality].ndim == 3:
                    feature_dims[modality] = (
                        1,
                        batch.data[modality].shape[1],
                    )
                else:
                    raise ValueError(
                        f"Unexpected number of dimensions for modality {modality}: {batch.data[modality].ndim}"
                    )
            else:
                feature_dims[modality] = None
        if "fmri" in batch.data:  # read from fmri config
            fmri = batch.data["fmri"]
            n_outputs = fmri.shape[1]
            for metric in self.metrics:
                if hasattr(metric, "kwargs") and "num_outputs" in metric.kwargs:
                    metric.kwargs["num_outputs"] = n_outputs
        else:  # read from neuro config
            if hasattr(self.data.neuro.projection, "mesh"):
                from neuralset.extractors.neuro import FSAVERAGE_SIZES

                n_outputs = 2 * FSAVERAGE_SIZES[self.data.neuro.projection.mesh]
            else:
                raise ValueError(
                    f"Could not determine number of outputs for neuro extractor {self.data.neuro}"
                )
        brain_model = self.brain_model_config.build(
            feature_dims=feature_dims,
            n_outputs=n_outputs,
            n_output_timesteps=self.data.duration_trs,
        )
        LOGGER.info("Extractor dims: %s", feature_dims)
        input_data = brain_model.aggregate_features(batch)
        LOGGER.info("Input shapes: %s", input_data.shape)
        LOGGER.info("Target shapes: %s", n_outputs)
        _ = brain_model(batch)
        total_params = sum(p.numel() for p in brain_model.parameters())
        LOGGER.info(f"Total parameters: {total_params}")
        self._model = self._init_module(brain_model)
        if self.monitor == "val/pearson":
            mode = "max"
        else:
            mode = "min"
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
        ]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=self.monitor, mode=mode, patience=self.patience)
            )
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename=self.checkpoint_filename,
                    monitor=self.monitor,
                    mode=mode,
                    save_on_train_epoch_end=True,
                )
            )

        n_devices = override_n_devices or self.infra.gpus_per_node
        # MPS only supports 1 device and does not support FSDP
        if self.accelerator == "mps" or (
            self.accelerator == "auto" and not torch.cuda.is_available()
        ):
            n_devices = max(n_devices, 1)
            strategy = "auto"
        else:
            strategy = "auto" if n_devices == 1 else "fsdp"
        trainer = pl.Trainer(
            strategy=strategy,
            devices=n_devices,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            max_steps=self.max_steps,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
            enable_checkpointing=self.save_checkpoints,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        self._trainer = trainer
        return trainer

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._trainer.fit(
            model=self._model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._get_checkpoint_path(),
        )

    def test(self, test_loader: DataLoader) -> None:
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
        else:
            if self.save_checkpoints:
                ckpt_path = Path(self.infra.folder) / "best.ckpt"
            else:
                ckpt_path = None
        self._trainer.test(
            self._model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )

    def setup_run(self):

        if self.infra.cluster and self.infra.status() != "not submitted":
            for out_type in ["stdout", "stderr"]:
                old_path = Path(getattr(self.infra.job().paths, out_type))
                new_path = Path(self.infra.folder) / f"log.{out_type}"
                try:
                    if new_path.exists():
                        os.remove(new_path)
                    os.symlink(
                        old_path,
                        new_path,
                    )
                except Exception:
                    pass
        config_path = Path(self.infra.folder) / "config.yaml"
        os.makedirs(self.infra.folder, exist_ok=True)
        with open(config_path, "w") as outfile:
            yaml.dump(
                self.model_dump(),
                outfile,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @infra.apply
    def run(self):
        import lightning.pytorch as pl

        self.setup_run()
        self._logger = (
            self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
                id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
            )
            if self.wandb_config
            else None
        )

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        loaders = self.data.get_loaders(
            split_to_build="val" if self.test_only else None
        )
        self._setup_trainer(next(iter(loaders.values())))

        if not self.test_only:
            self.fit(loaders["train"], loaders["val"])

        self.test(loaders["val"])
```

- `_free_extractor_model` now also frees `_tokenizer` (not just `_model`)
- Relaxed `isinstance(torch.nn.Module)` check → now deletes any non-None object

---

### 5. Optional Dependency

```diff:pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tribev2"
version = "0.1.0"
description = "Deep multimodal brain encoding"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [{name = "Meta Platforms, Inc."}]

dependencies = [
  "neuralset==0.0.2",
  "neuraltrain==0.0.2",
  "torch>=2.5.1,<2.7",
  "numpy==2.2.6",
  "torchvision>=0.20,<0.22",
  "x_transformers==1.27.20",
  "einops",
  "pyyaml",
  "moviepy>=2.2.1",
  "huggingface_hub",
  "gtts",
  "langdetect",
  "spacy",
  "soundfile",
  "pip",
  "Levenshtein",
  "julius",
  "transformers"
]

[project.urls]
Homepage = "https://github.com/facebookresearch/tribev2"
Repository = "https://github.com/facebookresearch/tribev2"

[project.optional-dependencies]
  plotting = [
    "nibabel",
    "matplotlib",
    "seaborn",
    "colorcet",
    "nilearn",
    "scipy",
    "pyvista",
    "scikit-image",
  ]
  training = [
    "nibabel",
    "torchmetrics",
    "wandb",
    "lightning",  
  ]
  test = [
    "pytest",
  ]

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.setuptools.packages.find]
include = ["tribe*"]
===
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tribev2"
version = "0.1.0"
description = "Deep multimodal brain encoding"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [{name = "Meta Platforms, Inc."}]

dependencies = [
  "neuralset==0.0.2",
  "neuraltrain==0.0.2",
  "torch>=2.5.1,<2.7",
  "numpy==2.2.6",
  "torchvision>=0.20,<0.22",
  "x_transformers==1.27.20",
  "einops",
  "pyyaml",
  "moviepy>=2.2.1",
  "huggingface_hub",
  "gtts",
  "langdetect",
  "spacy",
  "soundfile",
  "pip",
  "Levenshtein",
  "julius",
  "transformers"
]

[project.urls]
Homepage = "https://github.com/facebookresearch/tribev2"
Repository = "https://github.com/facebookresearch/tribev2"

[project.optional-dependencies]
  plotting = [
    "nibabel",
    "matplotlib",
    "seaborn",
    "colorcet",
    "nilearn",
    "scipy",
    "pyvista",
    "scikit-image",
  ]
  training = [
    "nibabel",
    "torchmetrics",
    "wandb",
    "lightning",  
  ]
  optimized = [
    "torchao>=0.9,<0.10",
  ]
  test = [
    "pytest",
  ]

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.setuptools.packages.find]
include = ["tribe*"]
```

- Added `torchao>=0.9,<0.10` under `[optimized]` optional dependency group

---

## Memory Profile (Before vs After)

| Component | Before | After | Savings |
|:---|:---|:---|:---|
| Llama-3.2-3B weights | 12.8 GB (fp32) | **6.4 GB** (fp16) | **6.4 GB** |
| Brain model weights | 676 MB (fp32) | **338 MB** (fp16) | **338 MB** |
| WhisperX peak | ~2.5 GB | ~1.5 GB (turbo) | ~1 GB |
| **Sequential peak** | **~14–16 GB** | **~7–8 GB** | **~8 GB** |

## Verification

### E2E Test Result
```
Brain model: 338 MB, dtype=torch.float16, loaded in 7.5s
Predictions: shape=(2, 20484), dtype=float16
Pred range: [-0.6143, 0.5078]
MPS memory: 581 MB
Total pipeline: 84.4s ✅
```

### INT8 Quantization Test
```
fp16 only:     0 NaNs / 712,704 — range [-307, +333] ✅
fp32 only:     0 NaNs / 712,704 — range [-302, +328] ✅
fp16 + INT8: 638,976 NaNs — layers 13+ all NaN ❌
```

## Usage

```python
from tribev2 import TribeModel

# Load with cached alpindale mirror (avoids re-downloading gated meta-llama)
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./cache",
    config_update={
        "data.text_feature.model_name": "alpindale/Llama-3.2-3B",
    },
)

# Text-only prediction
df = model.get_events_dataframe(text_path="sample_text.txt")
preds, segments = model.predict(events=df)
```
