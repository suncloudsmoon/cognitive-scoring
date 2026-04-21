# The Ad-Project HuggingFace Space

## Overview

The `ad-project/` folder contains a Gradio application originally hosted as a HuggingFace Space. It was designed to measure the "effectiveness of an Ad using tribev2."

This project heavily influenced our current analytical implementation. Reviewing it validated our approach and provided optimizations for scoring.

## What It Does

The Gradio App (`app.py` and `brain_regions.py`) takes an audio or video advertisement, processes it through TRIBE v2, and outputs an "Ad Effectiveness Score" based on predicted brain activity.

### The Pipeline
1. Takes user input (Video or Audio).
2. Runs inference using a cached or loaded TRIBE v2 model to get vertex predictions.
3. Groups vertices into **7 specific Advertising categories** using the Destrieux atlas.
4. Applies Z-score and Sigmoid normalization.
5. Averages the categories to create a single "Overall Ad Effectiveness Score."
6. Displays the result using interactive Plotly radar charts and timelines.

## The 7 Ad-Specific Categories

Unlike our 10 general-purpose neuroscientific groups in `brain_states.py`, `brain_regions.py` maps Destrieux labels specifically to advertising-relevant concepts:

| Ad Category | Brain Regions Used | Why It Matters for Ads |
|-------------|--------------------|------------------------|
| **Emotional Engagement** | Insula, orbitofrontal, ACC, subcallosal | Does the ad evoke an emotional response? (Drives sharing and intent) |
| **Visual Attention** | Calcarine, cuneus, lingual, occipital | Are people actually watching the visuals? |
| **Auditory Processing**| Superior temporal, Heschl's | Is the voiceover/music being processed clearly? |
| **Memory Encoding** | Parahippocampal, fusiform, temporal pole | Will the viewer remember the ad/brand later? |
| **Reward / Motivation**| Rectus, frontal pole, mid-post cingulate | Does the ad drive desire or intent to act? |
| **Language Comp.** | Broca's, angular, supramarginal | Is the message being understood? |
| **Social Cognition** | Superior frontal, precuneus, posterior cingulate | How does the viewer respond to faces/people in the ad? |

## What We Learned and Adopted

Studying this codebase revealed several best practices which we integrated into `tribev2/brain_states.py`:

**1. Z-Score + Sigmoid Normalization**
Our initial `brain_states.py` used raw heuristics (e.g., simply dividing frontoparietal by visual activation). The ad-project used a superior global Z-score coupled with a sigmoid transform (`1 / (1 + exp(-z))`). This cleanly converts raw BOLD predictions into comparable `0.0` to `1.0` scores centered precisely at `0.5`, making thresholds intuitive.

**2. Plotly Visualizations**
The radar chart visualization in the Gradio app is excellent for communicating complex high-dimensional brain data to laypeople. We extracted that logic and included it directly in `brain_states.py` (`create_radar_chart`, `create_timeline_chart`).

## Key Differences & Synergy

- **Use Case:** The Ad-Project is highly specialized for scoring commercials. Our `brain_states.py` is general-purpose (general emotion, learning, etc.).
- **Missing Features:** The Ad-Project only uses absolute scoring heuristics. It lacks the **Reference Classification** (Pearson correlation against known stimuli) and **Differential Mapping** that our `BrainStateProfiler` provides.

By combining the Ad-Project's robust normalization mathematical framework with our empirical reference profiling, we achieved the current state-of-the-art analysis pipeline demonstrated in `brain_state_demo.ipynb`.
