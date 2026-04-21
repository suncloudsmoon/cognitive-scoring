# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mac-local configuration for running TRIBE v2 on Apple Silicon (MPS).

Uses smaller models that fit in 16 GB unified memory:
- Text: Qwen/Qwen3-0.6B  (~2.5 GB fp32)
- Video: facebook/vjepa2-vitl-fpc64-256  (~1.5 GB fp32)
- Audio: facebook/w2v-bert-2.0  (~2.5 GB fp32)

Extractors run sequentially (each freed before the next loads),
keeping peak memory at roughly 7–8 GB.
"""

import copy
import os
from pathlib import Path

from exca import ConfDict

from .configs import mini_config

# Local paths – override via environment or edit here
_BASEDIR = os.getenv("SAVEPATH", os.path.join(os.getcwd(), "output"))
_CACHEDIR = os.path.join(_BASEDIR, "cache")

for _p in [_BASEDIR, _CACHEDIR]:
    Path(_p).mkdir(parents=True, exist_ok=True)

mac_config = ConfDict(copy.deepcopy(mini_config))
mac_config.update(
    {
        # ── Hardware / infra ──────────────────────────────────────────────
        "infra.cluster": None,
        "infra.workdir": None,
        "infra.folder": os.path.join(_BASEDIR, "results", "mac_local"),
        "infra.gpus_per_node": 1,
        "infra.mode": "force",
        "accelerator": "auto",
        # ── Data / memory ─────────────────────────────────────────────────
        "data.num_workers": 0,  # MPS + multiprocessing can deadlock
        "data.batch_size": 2,  # conservative for 16 GB
        # ── Extractors: run locally, no Slurm ─────────────────────────────
        "data.text_feature.infra.folder": _CACHEDIR,
        "data.text_feature.infra.cluster": None,
        "data.audio_feature.infra.folder": _CACHEDIR,
        "data.audio_feature.infra.cluster": None,
        "data.video_feature.infra.folder": _CACHEDIR,
        "data.video_feature.infra.cluster": None,
        "data.video_feature.image.infra.folder": None,
        "data.video_feature.image.infra.cluster": None,
        "data.neuro.infra.folder": _CACHEDIR,
        "data.neuro.infra.cluster": None,
        # ── Logging / checkpointing ───────────────────────────────────────
        "wandb_config": None,
        "save_checkpoints": False,
        "enable_progress_bar": True,
        # ── Short run for testing ─────────────────────────────────────────
        "n_epochs": 3,
    }
)


if __name__ == "__main__":
    from ..main import TribeExperiment  # noqa: E402

    folder = mac_config["infra"]["folder"]
    if os.path.exists(folder):
        import shutil

        shutil.rmtree(folder)

    xp = TribeExperiment(**mac_config)
    xp.infra.clear_job()
    xp.run()
