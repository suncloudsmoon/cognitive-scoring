# TRIBE v2 Project Wiki

This wiki documents everything about our TRIBE v2 project — the model, the modifications we made to run it on Apple Silicon, and the brain state analysis tools we built on top of it.

## Who Is This For?

This documentation is written so that **another LLM or developer** can read these files and fully understand:
- What TRIBE v2 is and how it works
- What every file in this project does
- How all the pieces fit together
- How to modify or extend any part of the system

## Wiki Pages

Read these in order:

| # | File | Topic |
|---|------|-------|
| 1 | [01_what_is_tribev2.md](./01_what_is_tribev2.md) | What TRIBE v2 is, what problem it solves, what it outputs |
| 2 | [02_architecture.md](./02_architecture.md) | How the model works internally — the full pipeline from input to brain prediction |
| 3 | [03_apple_silicon.md](./03_apple_silicon.md) | All the modifications made to run on Mac Mini M4 with MPS |
| 4 | [04_inference_pipeline.md](./04_inference_pipeline.md) | How to actually use the model — TribeModel, events, predictions |
| 5 | [05_brain_atlas.md](./05_brain_atlas.md) | The Destrieux atlas, brain regions, and how vertices map to named areas |
| 6 | [06_brain_states.md](./06_brain_states.md) | The brain_states.py module — scoring, classification, visualization |
| 7 | [07_ad_project.md](./07_ad_project.md) | The ad-project HuggingFace space and how it uses TRIBE v2 |
| 8 | [08_file_reference.md](./08_file_reference.md) | Every file in the project and what it does |
| 9 | [09_how_to_extend.md](./09_how_to_extend.md) | Guide for adding new features, new brain states, new scoring categories |
| 10 | [10_api_reference.md](./10_api_reference.md) | **BrainAPI** — the simple, two-line API for text → brain-state analysis |

## Key Concepts in One Paragraph

TRIBE v2 takes media (video, audio, or text) and predicts what a human brain would do if someone experienced that media. The output is an array of ~20,000 numbers representing activity at different points on the brain's surface. We map those points to named brain regions using the Destrieux atlas, then compute scores for things like "emotional engagement," "learning readiness," and "attention." This lets you answer questions like "does this text activate the brain's reward centers?" or "which parts of the brain respond differently to happy vs. sad content?"
