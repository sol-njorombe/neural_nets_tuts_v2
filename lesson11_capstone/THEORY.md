# Lesson 11 — Theory: running a real project

The earlier lessons each had a clean dataset, a fixed task and tests that said
when you were done. A real project has none of those. Most of the work is
deciding what to predict, getting honest data, and knowing whether the result is
any good. The network is often the easy part.

This note is the workflow. Each capstone brief in the README follows it.

## 1. Write the task down as inputs → outputs

One sentence, before any code: "Given **X**, predict **Y**, so that **someone**
can **do something**."

- "Given an unedited photo and its camera settings, predict the 12 Lightroom
  slider values I would choose, so I can start editing from a good first pass."
- "Given the last three letters, predict the next letter, so I can generate new
  names."

If you cannot write the sentence, the project is not ready for a model.

## 2. Choose the metric before the model

The loss trains the model; the **metric** decides whether it is useful. Pick one
a person cares about and can interpret:

| Task | Training loss | Metric to report |
|---|---|---|
| Classification | cross-entropy | accuracy, or accuracy on the class you care about |
| Regression (sliders) | MSE on scaled targets | mean absolute error per output, in the original units |
| Generation | cross-entropy | does a person find the samples plausible? |

## 3. Split honestly

Lesson 7 §1, the most common way real projects fool themselves. Split by
whatever makes examples similar: by shoot for photos, by patient for medical
records, by user for app data, by time for anything that changes (train on the
past, validate on the future). Then do not look at the validation set while
deciding what to try next more than you must, and keep a test set you look at once.

## 4. Build a baseline first

A baseline is the simplest thing that could work:

- always predict the most common class, or the average value;
- a hand-written rule;
- a non-neural model: logistic regression, or gradient boosting (scikit-learn's
  `HistGradientBoostingRegressor`), which is often excellent on tabular data.

If the neural network does not clearly beat the baseline on validation data, it
is not earning its complexity. Surprisingly often, the baseline wins.

## 5. Start small, then grow

1. **Overfit one batch.** Train on 10 examples until the loss is near zero. If
   it cannot, there is a bug (lessons 2–6 tell you where to look).
2. **Train on everything** with a small model; read the curves (lesson 7 §7).
3. **Grow** only what the curves say is limiting: more capacity if
   underfitting, more data or regularisation if overfitting.

## 6. Look at the mistakes

Sort validation examples by how wrong the model is, and look at the worst 20.
They usually reveal a problem in the data (wrong labels, duplicates, an input the
model cannot possibly use) before they reveal a problem in the model.

## 7. Decide when it is good enough

Write down, before you start, what result would make the project worth using.
"Within 0.3 stops of my exposure choice on 80% of photos" is a target; "as good
as possible" is not. Stop when you hit it, or when the last three ideas did not
move the validation metric.

## Checklist

- [ ] The task as one sentence: inputs, outputs, who uses it
- [ ] A metric in units a person understands
- [ ] Train / validation / test split by group, with no leakage
- [ ] A baseline and its validation score
- [ ] A small model that can overfit one batch
- [ ] Training curves read and diagnosed
- [ ] The worst 20 validation mistakes looked at
- [ ] A target, written down in advance, and the final test score against it
