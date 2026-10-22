# Lesson 11 — Capstone

**Read [THEORY.md](THEORY.md) first:** it is the workflow every brief below follows.

There are no stubs or tests in this lesson. Choose one brief, or bring your own
problem, and work through the THEORY checklist. Each brief says which earlier
lessons it draws on.

---

## A. Learn a photographer's editing style

**The task.** Given an unedited photo, predict the Lightroom develop settings the
photographer would apply, so that a batch of new photos arrives with a good first
pass already in place.

**Lessons used:** 7 (splits, diagnosis), 9 (PyTorch), 10 (pretrained backbone,
regression head).

**1. Collect pairs.** Each training example is (unedited photo, the settings the
photographer chose). Lightroom stores develop settings in two places:

- **XMP sidecar files** (or XMP embedded in DNG/JPEG), written when "Automatically
  write changes into XMP" is on, or with *Metadata → Save Metadata to File*. The
  basic sliders are attributes in the `crs:` namespace, for example
  `crs:Exposure2012`, `crs:Contrast2012`, `crs:Highlights2012`, `crs:Shadows2012`,
  `crs:Whites2012`, `crs:Blacks2012`, `crs:Texture`, `crs:Clarity2012`,
  `crs:Dehaze`, `crs:Vibrance`, `crs:Saturation`, `crs:Temperature` and `crs:Tint`.
- **The catalog** (`.lrcat`), which is a SQLite database. Reading it directly is
  possible but its layout is undocumented; work on a copy, never the live catalog.

The model's input must be the photo **before** editing: the camera's JPEG, or a
preview rendered with default settings. Training on edited exports would teach
it to predict settings from photos that already have them applied.

Note that white balance behaves differently for raw files (Temperature in Kelvin)
and JPEGs (a relative shift). Train on one kind, or treat them separately.

**2. Inputs.** Downsize each photo (for example to 224×224) and run it through a
pretrained backbone once (lesson 10 §4, §7), saving the features. Add simple
camera metadata as extra inputs: ISO, shutter speed, aperture, and as-shot white
balance.

**3. Targets.** Choose the sliders to predict (start with 6–12). Standardise each:
subtract its mean and divide by its standard deviation, so Temperature (thousands)
and Exposure (a few stops) count equally in the loss.

**4. Split by shoot.** Photos from one shoot look alike and are edited alike. If
the same shoot is in both training and validation, the score is inflated.
Hold out whole shoots.

**5. Baseline.** Predict the photographer's average setting for each slider (or the
average per camera). Report the validation mean absolute error per slider in its
original units. The network must beat this.

**6. Model.** A small MLP head on the frozen features, MSE loss on standardised
targets, Adam. Read the curves; fine-tune the backbone only if there is plenty of data.

**7. Look at the results, not just the numbers.** Different slider combinations
can produce nearly the same look (raising Exposure or raising Whites both
brighten), so a prediction can be "wrong" in slider space and right to the eye.
Apply predictions to some validation photos and compare side by side with the
real edits.

**8. Write back.** Write predicted values into XMP sidecars, or apply them with a
Lightroom Classic plugin (Lua SDK). Keep the predictions as a starting point the
photographer can adjust.

**Out of scope for a first version:** crops, local masks, and per-series
consistency (photographers often edit a whole sequence to match).

---

## B. A reinforcement learning agent

**The task.** A dot on a corridor of 7 cells can step left or right. Reaching the
right end gives +1, the left end −1, and each step costs 0.01. Nobody says which
move is right; the agent learns from reward alone.

**Lessons used:** 6 (softmax), 8 or 9 (autograd), 7 (optimisers).

**Policy:** a network from position (one-hot, 7 inputs) to two action logits.
Softmax gives probabilities; sample an action.

**Learning rule (REINFORCE):** after each episode, for every step taken, use the
action that was actually taken as if it were the correct label, weighted by the
total reward that followed it:

```
supervised (lesson 6):   ∂L/∂logits = (P − one_hot(correct label)) / n
REINFORCE:               ∂L/∂logits = (P − one_hot(action taken)) · G / n

G is the total reward that followed that step. Then step downhill as usual.
```

Good outcomes make the actions that led to them more likely; bad outcomes, less
likely. Subtracting the average return as a baseline makes learning much steadier.

**Stretch:** install `gymnasium` and balance CartPole. Value networks, self-play
and tree search on top of this are the outline of AlphaGo.

---

## C. A character-level name generator

**The task.** Given the previous 3 characters, predict the next one; then generate
new names one letter at a time.

**Lessons used:** 6 (softmax, cross-entropy), 9 (PyTorch).

**Data:** any list of names or words, one per line. Pad each with a start/end
marker, e.g. `...emma.`, and make one training example per position.

**Model:** an **embedding** (`nn.Embedding(27, 10)`: a learned vector per
character, lesson 10's "features" for letters), concatenated for the 3 context
characters, then a hidden layer and a 27-way softmax.

**Sampling:** start from `...`, sample the next character from the softmax,
shift, repeat until `.`.

This is a direct ancestor of an LLM: replace 3 characters with thousands of
tokens and the hidden layer with attention layers, and the training loop is
unchanged. Andrej Karpathy's *makemore* and *nanoGPT* lectures continue from here.

---

## D. Your own tabular data

**The task.** Take something you already record as rows of numbers (fitness logs,
energy use, project metrics) and predict one column from the others.

**Lessons used:** 7 (everything), 9.

Follow the THEORY checklist exactly, and compare three models on the same split:
the average baseline, scikit-learn's `HistGradientBoostingRegressor` (or
`...Classifier`), and a small PyTorch MLP. On small tabular datasets gradient
boosting often wins; seeing that first-hand is part of knowing when a neural
network is the right tool.
