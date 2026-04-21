# Project File Reference

This page lists every significant file in the project and explains its purpose.

## Root Directory

| File/Folder | Description |
|-------------|-------------|
| `brain_state_demo.ipynb` | The primary interactive notebook demonstrating our custom brain state analysis (ROI mapping, differential profiling, classification, and Plotly charts). |
| `tribe_demo.ipynb` | The original demo notebook provided by Meta, updated by our scripts to run on local Apple Silicon hardware. Validates base model functionality. |
| `cache/` | Directory where intermediate files are stored. Contains extracted features (`<hash>` folders) and saved brain profiles (`brain_states/`). |
| `stimuli/` | Contains `.txt` files (`happy.txt`, `sad.txt`, etc.) containing paragraphs of text used by `BrainStateProfiler` to generate reference brain patterns. |
| `scripts/` | Contains utility scripts, notably `update_demo_notebook.py` which patches the original Meta notebook. |
| `ad-project/` | Contains the HuggingFace space Gradio application for Ad Scoring (`app.py`, `brain_regions.py`). We studied this code to improve our own scoring normalization. |

## The Core Module: `tribev2/`

This is the main Python package. It contains the model architecture, data loading pipelines, and our custom analysis module.

| File | Description |
|------|-------------|
| `__init__.py` | Package init. Crucially, sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and imports `_mps_compat.py` immediately to ensure patches apply before PyTorch starts. Exports `TribeModel`, `BrainAPI`, `BrainResult`. |
| `_mps_compat.py` | Contains the monkey-patches applied to the `neuralset` library to force feature extractors (V-JEPA, Wav2Vec) onto the `mps` device while keeping LLaMA on `cpu`, handling float16 quantization. |
| `api.py` | **High-Level API.** Contains `BrainAPI` (the simple two-line interface to the full pipeline) and `BrainResult` (structured output dataclass). See `10_api_reference.md`. |
| `server.py` | **HTTP API Server.** FastAPI-based REST API wrapping `BrainAPI`. Provides `POST /analyze`, `POST /compare`, `GET /health`, `GET /regions` endpoints. Run with `python -c "from tribev2.server import main; main()"`. |
| `main.py` | Defines the heavy-lifting classes: `Data` (manages DataLoaders and event alignment) and `TribeExperiment` (the PyTorch Lightning training/test loop wrapper). Contains memory-freeing logic (`_free_extractor_model`). |
| `model.py` | Architecture file. Defines the `FmriEncoderModel` (the Fusion Transformer) that takes the extracted features, combines them, processes them via attention, and outputs the final 20,484-vertex predictions. |
| `demo_utils.py` | Inference-time wrapper. Defines `TribeModel.from_pretrained()` to load models from HuggingFace, and `get_events_dataframe()` which handles the Text-to-Speech-to-Transcription pipeline for text inputs. |
| `brain_states.py` | **Our Custom Module.** Contains `BrainAtlas` (Destrieux mapping), Normalized Scoring logic, `BrainStateProfiler` (building reference `.npy` files from the `stimuli/` dir), and Plotly visualization functions. |
| `utils_fmri.py` | Contains `TribeSurfaceProjector`, the logic that projects volumetric voxel data onto the fsaverage5 cortical mesh. |
| `eventstransforms.py` | Contains data manipulation steps to convert raw video/audio/text into aligned DataFrames using libraries like WhisperX. |
| `pl_module.py` | PyTorch Lightning module definition (`BrainModule`) managing the loss function and optimization steps used during the original training. |

## Visualization: `tribev2/plotting/`

Contains code for generating the 3D brain surface heatmaps.

| File | Description |
|------|-------------|
| `base.py` | Base classes for plotting. |
| `cortical.py` | Brain visualization using `nilearn`. |
| `cortical_pv.py` | Brain visualization using `pyvista` (3D interactive/rendered plotting). This is the default `PlotBrain` class used in the notebooks. |
| `utils.py` | Matplotlib and color-mapping utilities. |
