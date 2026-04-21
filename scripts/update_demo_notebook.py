#!/usr/bin/env python3
"""
Script to update tribe_demo.ipynb for local Mac M4 MPS usage.

Changes:
1. Updates the setup markdown cell for local Mac usage
2. Replaces the Colab install cell with a no-op / local instructions
3. Updates model loading to use HuggingFace hub (facebook/tribev2)
4. Clears all cell outputs (removes stale errors and huge embedded images)
"""

import json
import sys
from pathlib import Path


def update_notebook(notebook_path: Path) -> None:
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    cells = nb["cells"]

    for cell in cells:
        # Clear all outputs
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # --- Cell 0 (index 0): Update intro markdown ---
    # Find the intro markdown cell and update it for MPS
    for cell in cells:
        if cell["cell_type"] == "markdown" and "TRIBE v2 Demo" in "".join(cell["source"]):
            cell["source"] = [
                "# TRIBE v2 Demo: Predicting Brain Responses to Naturalistic Stimuli\n",
                "\n",
                "[TRIBE v2](https://github.com/facebookresearch/tribev2) is a deep multimodal brain encoding model that predicts **fMRI brain responses** to naturalistic stimuli — video, audio, and text.\n",
                "\n",
                "It combines state-of-the-art feature extractors — **LLaMA 3.2** (text), **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) — into a unified Transformer that maps multimodal representations onto the cortical surface (**fsaverage5**, ~20k vertices).\n",
                "\n",
                "In this notebook, we will:\n",
                "1. Load a pretrained TRIBE v2 model from HuggingFace\n",
                "2. Predict brain responses to a **video** clip\n",
                "3. Predict brain responses to **audio** generated from text\n",
                "4. Visualize the predicted activity on a 3D brain surface\n",
                "\n",
                "> **Note (Mac M4 / MPS):** This notebook has been adapted for local execution on Apple Silicon (M4) with MPS acceleration. The `tribev2` package includes monkey-patching for MPS device support. Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in your environment (this is done automatically by the `tribev2` package).\n",
            ]
            break

    # --- Cell 1 (index 1): Update setup markdown ---
    for cell in cells:
        if cell["cell_type"] == "markdown" and "Setup" in "".join(cell["source"]) and "Colab" in "".join(cell["source"]):
            cell["source"] = [
                "## Setup\n",
                "\n",
                "### For local Mac M4 users:\n",
                "1. Ensure you have the `tribev2` package installed in your virtual environment: `pip install -e \".[plotting]\"`\n",
                "2. The MPS compatibility patches are applied automatically when importing `tribev2`\n",
                "3. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically\n",
                "\n",
                "### For Colab users:\n",
                "1. Activate the GPU (Menu > Runtime > Change runtime)\n",
                "2. Run the install command below\n",
                "3. Restart your environment for the new packages to be taken into account\n",
            ]
            break

    # --- Cell 2 (index ~2): Update the install cell ---
    for cell in cells:
        if cell["cell_type"] == "code" and "uv pip install" in "".join(cell["source"]):
            cell["source"] = [
                "# For Colab users only — skip this cell if running locally\n",
                "# !uv pip install \"tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git\"\n",
                "\n",
                "# For local Mac M4 users, ensure tribev2 is installed in your venv:\n",
                "# pip install -e \".[plotting]\"\n",
                "print(\"Setup complete. If running locally, ensure tribev2 is installed in your venv.\")\n",
            ]
            break

    # --- Cell 3 (index ~3): Update model loading markdown ---
    for cell in cells:
        if cell["cell_type"] == "markdown" and "Loading the model" in "".join(cell["source"]):
            cell["source"] = [
                "## Loading the model\n",
                "\n",
                "We load TRIBE v2 model from [HuggingFace Hub](https://huggingface.co/facebook/tribev2). On the first run, this downloads the model checkpoint and config (~1 GB). Subsequent runs use the cached version.\n",
                "\n",
                "We also initialize a `PlotBrain` object for 3D brain surface visualization using the **fsaverage5** mesh.\n",
                "\n",
                "> **Mac M4 Note:** The model will automatically load on the MPS device thanks to the monkey-patching in `tribev2/_mps_compat.py`. Feature extractors will resolve `device='auto'` to `mps` when CUDA is unavailable.\n",
            ]
            break

    # --- Cell 4 (index ~4): Update model loading code ---
    for cell in cells:
        if cell["cell_type"] == "code" and "TribeModel.from_pretrained" in "".join(cell["source"]):
            cell["source"] = [
                "from tribev2.demo_utils import TribeModel, download_file\n",
                "from tribev2.plotting import PlotBrain\n",
                "from pathlib import Path\n",
                "\n",
                "CACHE_FOLDER = Path(\"./cache\")\n",
                "\n",
                "# Load from HuggingFace Hub (recommended)\n",
                "model = TribeModel.from_pretrained(\n",
                "    \"facebook/tribev2\",\n",
                ")\n",
                "\n",
                "# Alternative: load from local weights\n",
                "# model = TribeModel.from_pretrained(\n",
                "#     \"/path/to/local/weights\",\n",
                "# )\n",
                "\n",
                "plotter = PlotBrain(mesh=\"fsaverage5\")\n",
            ]
            break

    # --- Update the "Run the model" markdown to remove Llama access note ---
    for cell in cells:
        if cell["cell_type"] == "markdown" and "Run the model" in "".join(cell["source"]) and "Llama" in "".join(cell["source"]):
            cell["source"] = [
                "### Run the model\n",
                "\n",
                "We feed the events dataframe to `model.predict()`, which extracts features for each modality, runs them through the Transformer, and returns predicted brain activity.\n",
                "\n",
                "> **Note:** On Mac M4, feature extraction (especially WhisperX for transcription) runs on CPU with `int8` quantization for best compatibility. The TRIBE model itself runs on MPS.\n",
                "\n",
                "The output `preds` has shape `(n_timesteps, n_vertices)` — one prediction per second of stimulus, with ~20k cortical vertices. The `segments` list contains the corresponding time segments with their associated events.\n",
            ]
            break

    # Write back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
        f.write("\n")

    print(f"✅ Updated {notebook_path}")
    print("   - Cleared all cell outputs")
    print("   - Updated setup instructions for Mac M4 / MPS")
    print("   - Updated model loading to use HuggingFace hub")
    print("   - Added MPS compatibility notes")


if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / "tribe_demo.ipynb"
    if not notebook_path.exists():
        print(f"❌ Notebook not found at {notebook_path}")
        sys.exit(1)
    update_notebook(notebook_path)
