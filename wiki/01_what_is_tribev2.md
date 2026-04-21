# What Is TRIBE v2?

## The One-Sentence Version

TRIBE v2 is a neural network from Meta AI that takes in media (video, audio, or text) and outputs a prediction of what the human brain's activity would look like on an fMRI scanner if someone were experiencing that media.

## The Problem It Solves

Neuroscience researchers use fMRI scanners to measure brain activity while people watch videos, listen to audio, or read text. But fMRI scanning is:
- **Expensive** — a single scanning session costs thousands of dollars
- **Slow** — you can only scan one person at a time, and sessions take hours
- **Noisy** — individual brain scans are very noisy; you need many subjects to get reliable patterns

TRIBE v2 replaces this with a computational prediction. Instead of putting someone in a scanner, you feed the stimulus (a video clip, an audio file, a piece of text) into the model, and it predicts what the brain scan *would have looked like*. Meta calls this **"in-silico neuroscience"** — running neuroscience experiments on a computer instead of in a lab.

## What Does "fMRI" Mean?

**fMRI** stands for **functional Magnetic Resonance Imaging**. It measures brain activity by detecting changes in blood flow. When a brain region is active, it uses more oxygen, which changes the magnetic properties of nearby blood. The fMRI scanner detects this change.

The specific signal is called **BOLD** (Blood-Oxygen-Level-Dependent). When you see "BOLD signal" or "fMRI activation" in this codebase, it means "how much blood flow (and therefore neural activity) is happening at that brain location."

Key properties of fMRI:
- **Temporal resolution**: ~1 second per measurement (called a "TR" — repetition time)
- **Spatial resolution**: measures thousands of points across the brain surface
- **Hemodynamic lag**: Blood flow changes happen about **5 seconds after** the neural event. TRIBE v2 compensates for this internally.

## What Does the Model Output?

TRIBE v2 outputs a **numpy array** with shape `(n_timesteps, 20484)`:

```
preds.shape = (n_timesteps, 20484)
              ↑                ↑
              |                └── 20,484 cortical surface vertices
              └── One row per second of stimulus (1 TR = 1 row)
```

### What Are "Vertices"?

The brain's outer surface (the cortex) is wrinkled and folded. To represent it computationally, you approximate it as a mesh of triangles — just like a 3D model in a video game. Each corner of a triangle is a **vertex**.

TRIBE v2 uses the **fsaverage5** mesh, which is a standard template from FreeSurfer (a neuroimaging software suite). It has:
- **10,242 vertices** for the left hemisphere
- **10,242 vertices** for the right hemisphere
- **20,484 total vertices**

The first 10,242 values in each row are the left hemisphere; the next 10,242 are the right hemisphere.

### What Does Each Number Mean?

Each number in the output array is a **predicted BOLD signal value** at that vertex at that time point. A higher value means more predicted neural activity at that location.

These values are relative, not absolute — what matters is the *pattern* of activation across the brain, not the specific numbers.

## How Was It Trained?

TRIBE v2 was trained on a massive dataset:
- **1,000+ hours** of fMRI recordings
- **720+ human subjects**
- People watched movies, listened to podcasts, read text
- The model learned to predict fMRI output from the input stimuli
- It uses **average-subject** mode by default — predicting the response of a "typical" brain rather than any specific individual

## The Three Modalities

TRIBE v2 handles three types of input simultaneously:

| Modality | Feature Extractor | What It Captures |
|----------|-------------------|------------------|
| **Video** | V-JEPA2 (ViT-G) | Visual scenes, motion, objects, faces |
| **Audio** | Wav2Vec-BERT 2.0 | Sounds, music, speech prosody |
| **Text** | LLaMA 3.2-3B | Language meaning, semantics, context |

When you feed a video, it extracts all three modalities automatically: the visual frames, the audio track, and the transcribed text. When you feed just text, it converts it to speech first (using Google TTS), then processes the generated audio and the original text.

## What Can You Do With the Output?

The raw output (20,484 numbers per timestep) is hard to interpret on its own. You need additional tools to make it useful:

1. **Visualize on brain surface** — Plot the values as a heatmap on the 3D brain shape (using the PlotBrain class)
2. **Map to brain regions** — Group the 20,484 vertices into named brain regions using an atlas (like Destrieux), then see which regions are most active
3. **Compare stimuli** — Feed in "happy" text and "sad" text, then compare the predicted brain patterns to see which regions distinguish them
4. **Classify brain states** — Build reference profiles for different emotional/cognitive states, then match new stimuli against them

All of these are implemented in this project. See the later wiki pages for details.

## Important Limitations

1. **Cortical only**: The output covers only the brain's outer surface (cortex). Subcortical structures like the amygdala, hippocampus, and basal ganglia are NOT in the output. These are some of the most important structures for emotion and memory.

2. **Average, not individual**: The model predicts what an "average" brain would do. It does not predict what any specific person's brain would do.

3. **Encoding, not decoding**: The model goes stimulus → brain pattern. It does NOT go brain pattern → "this person is happy." You have to build that interpretation layer yourself (which is what our brain_states.py module does).

4. **Not clinical**: Never use this for medical diagnosis or treatment decisions.
