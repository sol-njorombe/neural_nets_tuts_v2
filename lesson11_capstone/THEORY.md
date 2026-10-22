# Lesson 11 — Theory: running a real project

The earlier lessons each had a clean dataset, a fixed task and tests that said
when you were done. A real project has none of those. Most of the engineering work is
deciding what to predict, getting honest data, and knowing whether the result is
any good. The neural network architecture is often the easiest part.

This note outlines the end-to-end production workflow. Each capstone brief in the README follows it.

---

## 0. Learning Objectives

By the end of this capstone, you will be able to:

1. **Formulate Applied Problems**: Translate a real-world goal into a precise mathematical input $\to$ output contract with an interpretable domain metric.
2. **Design Leakage-Free Partitions**: Identify confounding duplicate groups (patients, sessions, shoots, temporal splits) and enforce group-based data splitting.
3. **Establish Meaningful Baselines**: Build simple heuristic or classical non-neural baselines (mean prediction, gradient boosted trees) that your network must convincingly beat.
4. **Execute Diagnostic-Driven Iteration**: Track train vs validation curves and analyze the worst-20 validation errors to diagnose data vs modeling failures.
5. **Critique Applied Machine Learning Work**: Evaluate projects against a professional engineering review rubric.

---

## 1. Write the task down as inputs → outputs

One sentence, before writing any code:
> "Given **X**, predict **Y**, so that **someone** can **do something**."

- *"Given an unedited photo and its camera metadata, predict the 12 Lightroom develop slider values I would choose, so I can start editing from a high-quality initial pass."*
- *"Given the last three characters, predict the next character, so I can generate novel domain-specific names."*

If you cannot formulate this sentence clearly, the project is not ready for machine learning.

---

## 2. Choose the metric before the model

The loss function trains the model via gradient descent; the **metric** determines whether the system delivers real-world utility.
Always select a metric expressed in interpretable domain units:

| Task Type | Optimization Loss | Evaluation Metric to Report |
|---|---|---|
| **Multi-class Classification** | Categorical Cross-Entropy | Top-1 accuracy, per-class F1 score, precision/recall |
| **Multi-output Regression** | Mean Squared Error (on normalized targets) | Mean Absolute Error (MAE) per output in **original raw units** (e.g. exposure stops, Kelvin) |
| **Generative Sequence Modeling** | Perplexity / Cross-Entropy | Human plausibility rate, distinct n-grams, diversity |

---

## 3. Split honestly

Recall Lesson 7 §1: data leakage is the most pervasive reason models succeed in notebooks and fail in production.
Always split data by the natural clustering group:
- **Photos / Creative Media**: Split by photoshoot / scene. Photos from the same shoot share lighting and subject matter.
- **Healthcare**: Split by patient. Scans from the same individual share anatomical features.
- **Time-Series / Telemetry**: Split chronologically (train on past, validate on future). Never randomly shuffle time series!
- **User Applications**: Split by user account / organization.

Never look at test set performance until the final candidate model is frozen.

---

## 4. Build a baseline first

A baseline is the simplest heuristic that could conceivably work:
- Predicting the dataset mean or majority class.
- A deterministic hand-coded rule.
- A classical non-deep learning model: Logistic Regression or Gradient Boosted Decision Trees (`HistGradientBoostingRegressor` in scikit-learn).

If a neural network fails to clearly outperform a simple gradient boosted tree on tabular data or a heuristic mean on regression, it does not justify its operational complexity. In industry, baselines frequently win on tabular tasks.

---

## 5. Start small, then grow

1. **Overfit a Single Batch**: Train on 10 examples until training loss reaches $\approx 0$. If a model cannot overfit 10 examples, you have a bug (gradient detachment, incorrect broadcasting, broken target reshaping).
2. **Train a Baseline Network on All Data**: Observe training vs validation trajectories using Lesson 7's diagnostic rubric.
3. **Targeted Iteration**:
   - If underfitting: expand width/depth, tune learning rate, switch to Adam.
   - If overfitting: collect more data, apply early stopping, add weight decay, or introduce dropout.

---

## 6. Inspect the Worst Mistakes

Sort validation examples by loss magnitude and examine the worst 20 predictions.
In real-world applications, extreme validation errors typically expose:
- Mislabeled ground-truth examples.
- Corrupted inputs or missing features.
- Severe ambiguities where two identical inputs received opposing human labels.

Fix data quality before tuning model hyperparameters.

---

## 7. Project Review Rubric

| Engineering Dimension | Production Standard (Exemplary) | Deficient (Needs Work) |
|---|---|---|
| **Problem Definition** | Clear inputs $\to$ outputs sentence with human action | Vague statement ("use deep learning on images") |
| **Metric Selection** | Unambiguous domain metric in real-world units | Only reporting standardized MSE loss |
| **Data Partitioning** | Grouped or temporal split with proven zero leakage | Random row split with near-duplicate leakage |
| **Baseline Benchmark** | Simple baseline scored on identical validation split | No baseline comparison |
| **Diagnostic Rigor** | Training curves plotted and systematically diagnosed | Random hyperparameter changes without curves |
| **Error Analysis** | Qualitative inspection of worst 20 validation errors | No inspection of failure modes |
| **Scientific Honesty** | Objective report of limitations and baseline comparisons | Reporting only best training run or cherry-picked cases |

---

## 8. Capstone Checklist

- [ ] One-sentence task contract: inputs $X$, outputs $Y$, user action
- [ ] Primary metric defined in real domain units
- [ ] Grouped train / validation / test splits verified without leakage
- [ ] Naive / classical baseline implemented and scored
- [ ] Single-batch overfit test passed ($L \to 0$)
- [ ] Learning curves logged and classified (*Diverging*, *Underfitting*, *Overfitting*, *Healthy*)
- [ ] Qualitative review of worst-20 validation errors
- [ ] Pre-established success criteria evaluated against final test split
