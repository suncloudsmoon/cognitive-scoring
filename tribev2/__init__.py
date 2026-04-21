# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TRIBE v2 — Predict brain activity from text, audio, and video.

TRIBE v2 is a deep multimodal brain encoding model from Meta AI that predicts
fMRI brain responses to naturalistic stimuli.  This package provides both a
Python API and an HTTP server for running inference and brain-state analysis.

Quick start::

    from tribev2 import BrainAPI

    api = BrainAPI.load()
    result = api.analyze("She opened the letter and tears of joy streamed down her face.")
    print(result.summary())

For video support, install with ``pip install cognitive-scoring[video]``.
For the HTTP server, install with ``pip install cognitive-scoring[server]``.
"""

import os

# Enable MPS fallback so unsupported Metal operations use CPU instead of crashing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tribev2._mps_compat  # noqa: F401  — patches neuralset for MPS before any extractor is created

from tribev2.api import BrainAPI, BrainResult
from tribev2.demo_utils import TribeModel

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("cognitive-scoring")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["TribeModel", "BrainAPI", "BrainResult", "__version__"]
