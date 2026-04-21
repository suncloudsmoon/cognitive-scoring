<div align="center">

# TRIBE v2

**A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience**

[![PyPI](https://img.shields.io/pypi/v/tribev2.svg)](https://pypi.org/project/tribev2/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)

📄 [Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) ▶️ [Demo](https://aidemos.atmeta.com/tribev2/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a deep multimodal brain encoding model from Meta AI that predicts fMRI brain responses to naturalistic stimuli. It maps text, audio, and video through a Fusion Transformer onto the **fsaverage5** cortical surface (~20k vertices).

## Quick Start

```bash
pip install tribev2
```

### Two-line API

```python
from tribev2 import BrainAPI

api = BrainAPI.load()
result = api.analyze("She opened the letter and tears of joy streamed down her face.")

print(result.valence)         # +0.031  (positive = happy)
print(result.learning)        # 0.577   (deeper cognitive processing)
print(result.attention)       # 0.600   (focused attention)
print(result.scores)          # {"prefrontal": 0.62, "temporal": 0.63, ...}
print(result.classification)  # [("happy", 0.85), ("calm", 0.32), ...]
print(result.summary())       # Full formatted output with bar charts
```

### HTTP API

```bash
pip install tribev2[server]
python -c "from tribev2.server import main; main()"
```

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "She opened the letter and tears of joy streamed down her face."}'
```

Interactive docs at `http://localhost:8000/docs`.

### Lower-level API

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(text_path="story.txt")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, 20484)
```

## Installation

```bash
# Core (text + audio inference)
pip install tribev2

# With video support (adds torchvision + moviepy)
pip install tribev2[video]

# With HTTP API server
pip install tribev2[server]

# With brain visualization (3D surface plots)
pip install tribev2[plotting]

# Everything except training
pip install tribev2[all]

# For development
pip install -e ".[all,test]"
```

### Optional extras

| Extra | What it adds | Use case |
|-------|-------------|----------|
| `video` | torchvision, moviepy | Video file input (`.mp4`, `.avi`, etc.) |
| `server` | FastAPI, uvicorn | HTTP API server |
| `plotting` | nilearn, pyvista, matplotlib | Brain surface heatmaps |
| `training` | lightning, wandb, torchmetrics | Model training from scratch |
| `optimized` | torchao | INT8 quantization |
| `all` | video + plotting + server + optimized | Everything |

## What You Get

### Region Scores (0–1)

Each text is scored on 10 functional brain region groups. A score of **0.5 = baseline**; above means more activation than average.

| Region | What it measures |
|--------|-----------------|
| `prefrontal` | Executive function, planning, decision-making |
| `reward_vmPFC` | Reward processing, positive affect |
| `anterior_cingulate` | Conflict monitoring, curiosity |
| `default_mode` | Self-referential thought, mind-wandering |
| `insula` | Emotional awareness, negative affect |
| `temporal` | Language comprehension, social cognition |
| `visual` | Visual processing |
| `attention_parietal` | Focused attention |
| `motor` | Sensorimotor processing |
| `fusiform_parahip` | Memory encoding, face/object recognition |

### Composite Scores

| Score | Formula | Meaning |
|-------|---------|---------|
| **Valence** | `reward − (insula + ACC) / 2` | Positive = happy, negative = sad |
| **Learning** | `(prefrontal + ACC + temporal) / 3` | Higher = deeper processing |
| **Attention** | `(parietal + prefrontal) / 2` | Higher = more focused |

## Performance

| Operation | First run | Cached |
|-----------|-----------|--------|
| Model loading | ~15s | ~15s |
| Text analysis | ~3–5 min | ~5–10s |
| Compare two texts | ~6–10 min | ~10–20s |

Feature extraction (V-JEPA, Wav2Vec, LLaMA 3.2) is the bottleneck. Results are cached by content hash; repeated analysis of the same text is instant.

## Training

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"

# Local test run
python -m tribev2.grids.test_run

# Grid search on Slurm
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project Structure

```
tribev2/
├── api.py               # BrainAPI: simple two-line interface
├── server.py            # FastAPI HTTP server
├── demo_utils.py        # TribeModel: model loading + inference
├── brain_states.py      # BrainAtlas, scoring, classification
├── model.py             # FmriEncoder: Fusion Transformer architecture
├── main.py              # Data + TribeExperiment pipeline
├── _mps_compat.py       # Apple Silicon MPS patches
├── eventstransforms.py  # Text/audio/video → events
├── plotting/            # Brain visualization backends
└── studies/             # Dataset definitions
```

## Caveats

- Predictions are **cortical surface only** — subcortical structures (amygdala, hippocampus, basal ganglia) are NOT represented.
- All cognitive/emotional labels are approximations based on cortical correlates.
- **NOT suitable for clinical diagnosis or treatment decisions.**

## Citation

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

Copyright © Meta Platforms, Inc. and affiliates. All rights reserved.

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0
International License](https://creativecommons.org/licenses/by-nc/4.0/legalcode) (CC BY-NC 4.0).

You may use, share, and adapt this material for **non-commercial purposes only**, provided you give appropriate credit, indicate if changes were made, and do not impose additional restrictions. See [LICENSE](LICENSE) for the full legal text.

> **Disclaimer of Warranties (§5):** This software is provided "AS-IS" and
> "AS-AVAILABLE" without any warranties of any kind, express or implied,
> including but not limited to warranties of merchantability, fitness for a
> particular purpose, or non-infringement. In no event shall the licensor be
> liable for any damages arising from use of this software.

**Modifications:** This repository contains modifications to the original TRIBE v2 codebase by Meta Platforms, Inc., including (among other things) a high-level Python API, an HTTP server, ROI-based brain-state scoring, and Apple Silicon compatibility patches. For a complete record of all changes, see the [git history](https://github.com/facebookresearch/tribev2/commits). These modifications are also licensed under CC BY-NC 4.0.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
