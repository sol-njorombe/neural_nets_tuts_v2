# Lesson 11 — Capstone Project

**Read [THEORY.md](THEORY.md) first:** it defines the engineering workflow that every project brief below follows.

There are no unit tests in this lesson. Choose one of the four briefs below (or bring a problem from your own work) and execute the project against the [THEORY Checklist](THEORY.md#8-capstone-checklist).

---

## Brief A: Learn a Photographer's Editing Style

### The Task
Given an unedited RAW or JPEG photo with camera metadata, predict the develop slider adjustments the photographer would apply in Adobe Lightroom, automating a high-quality initial pass across incoming photo batches.

**Relevant Lessons**: [Lesson 7 (Generalisation & Splits)](../lesson07_generalisation/), [Lesson 9 (PyTorch)](../lesson09_pytorch/), [Lesson 10 (Transfer Learning)](../lesson10_transfer/).

### Execution Steps
1. **Data Harvesting**: Export pairs of unedited photos and corresponding develop parameters. Lightroom stores parameters in XMP sidecars (`crs:Exposure2012`, `crs:Contrast2012`, `crs:Highlights2012`, `crs:Shadows2012`, `crs:Temperature`, `crs:Tint`).
2. **Feature Extraction**: Downsample images (e.g. $224 \times 224$) and pass through a pre-trained frozen vision backbone (e.g. ResNet-50 via `torchvision`), extracting a 2,048-dimensional feature embedding. Concatenate standardized EXIF metadata (ISO, aperture, shutter speed).
3. **Target Normalization**: Standardize target slider values ($\mu = 0, \sigma = 1$) so large numerical ranges (Temperature in Kelvin) and small ranges (Exposure in stops) contribute equally to MSE loss.
4. **Grouped Partitioning**: **Hold out entire shoots**. Never randomly split photos from the same session between train and validation.
5. **Baseline Comparison**: Implement a simple baseline predicting the photographer's historical mean slider settings.
6. **Model**: Multi-layer regression head trained with Adam and early stopping.
7. **Metric**: Report Mean Absolute Error (MAE) per slider in original photographic units (e.g. stops of exposure, Kelvin).

---

## Brief B: Reinforcement Learning Agent (REINFORCE)

### The Task
An autonomous agent navigates a 1D corridor of 7 discrete states. Moving right yields $+1.0$ at the terminal cell, moving left yields $-1.0$, and each transition incurs a step penalty of $-0.01$. The agent receives no supervised action labels; it must discover optimal policy dynamics purely from reward feedback.

**Relevant Lessons**: [Lesson 6 (Softmax)](../lesson06_layers/), [Lesson 8 or 9 (Autograd)](../lesson08_autograd/), [Lesson 7 (Optimisers)](../lesson07_generalisation/).

### Execution Steps
1. **Policy Network**: Parameterize a policy network mapping one-hot state vectors ($7$ inputs) to two action logits (left vs right). Apply softmax to sample actions.
2. **Policy Gradient (REINFORCE)**:
   $$\frac{\partial L}{\partial \text{logits}} = \frac{1}{n} (P - \text{one\_hot}(\text{action\_taken})) \cdot G$$
   where $G_t = \sum_{k=t}^T \gamma^{k-t} R_k$ is the discounted cumulative return following step $t$.
3. **Baseline Subtraction**: Subtract the rolling average return $b$ from $G$ to reduce gradient variance: $(G - b)$.
4. **Stretch Goal**: Generalize the policy to OpenAI `gymnasium` environments (e.g. CartPole-v1).

---

## Brief C: Character-Level Language Model

### The Task
Given a context window of the preceding 3 characters, predict the probability distribution over the next character; generate novel names or words autoregressively by sampling one letter at a time.

**Relevant Lessons**: [Lesson 6 (Softmax & Cross-Entropy)](../lesson06_layers/), [Lesson 9 (PyTorch)](../lesson09_pytorch/).

### Execution Steps
1. **Vocabulary & Dataset**: Build a token vocabulary from an input corpus (e.g. names list), mapping characters to indices plus a terminal delimiter `.`.
2. **Embedding Architecture**:
   - `nn.Embedding(vocab_size, embedding_dim)`: maps each of the 3 context characters to a learned vector.
   - Concatenate embeddings, feed through a hidden `nn.Linear` layer with `nn.ReLU()`, and project to 27 logits.
3. **Loss & Training**: Train using `nn.CrossEntropyLoss()` with Adam.
4. **Autoregressive Generation**: Seed with `...` and iteratively sample the next character distribution until the terminal token `.` is generated.

---

## Brief D: Real Tabular Data

### The Task
Extract an existing structured dataset from your own domain (fitness logs, server metrics, home energy consumption) and train a neural regression or classification model.

**Relevant Lessons**: [Lesson 7 (Generalisation & Splits)](../lesson07_generalisation/), [Lesson 9 (PyTorch)](../lesson09_pytorch/).

### Execution Steps
1. **Formulate Task**: Write the one-sentence contract: "Given $X$, predict $y$, so that user can take action $A$."
2. **Preprocessing**: One-hot encode categorical features; normalize numerical features.
3. **Rigorous Baselines**: Compare your neural network against:
   - Dummy mean / majority class baseline.
   - Scikit-learn's `HistGradientBoostingRegressor` / `Classifier`.
4. **Analysis**: Determine whether deep learning outperforms tree ensembles on tabular data.

---

## Self-Check Questions for Project Kickoff

1. **Utility Test**: What concrete action will a human or software system take based on your model's predictions? If this cannot be stated concisely, the project is ill-defined.
2. **Leakage Vulnerabilities**: Which examples in your dataset are correlated enough to cause data leakage? How does your partition strategy isolate them?
3. **Baseline Feasibility**: What is the simplest heuristic that could serve as a benchmark? What validation score do you expect it to achieve?
4. **Data Sizing & Pretraining**: How many labelled examples are available? Can you bootstrap feature extraction using a pre-trained backbone?
5. **Success Criteria**: What numerical performance metric would make this system valuable enough to deploy? Define this target threshold before training.

---

## Common Pitfalls & Troubleshooting

| Pitfall | Root Cause | Engineering Solution |
|---|---|---|
| Spending days on model architecture before having data | Focusing on modeling rather than data pipelines | Build the simplest baseline pipeline end-to-end on day 1 |
| Artificial validation scores due to leakage | Random row splitting across grouped entities | Split by shoot, patient, user, or time window |
| Reporting MSE on normalized target variables | Normalized loss is uninterpretable to stakeholders | Report MAE in original physical units (stops, Kelvin, dollars) |
| Declaring success based on training accuracy | Confusing memorisation with generalization | Always evaluate on untouched validation and test splits |
| Model cannot overfit 10 examples | Gradient detachment, shape corruption, or broken labels | Verify single-batch overfit test first before full training |
