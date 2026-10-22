# Lesson 7 — Teaching notes

This is the lesson that turns "I can train a network" into "I can tell whether it
is any good". It matters most for anyone who will go on to build real models.

## Learning objectives

By the end, the learner can:

1. Explain why validation data exists, and what leakage is, with an example.
2. Read train/validation curves and name overfitting, underfitting and divergence.
3. Explain why initialisation scale matters, and derive He initialisation's √(2/d_in).
4. Explain what momentum and Adam fix about plain SGD.
5. Implement early stopping that keeps the best weights.

## Suggested session (about 2.5 hours)

| Time | Activity |
|---|---|
| 15 min | The exam analogy; sketch the two curve shapes from THEORY §2 |
| 15 min | Variance argument for initialisation on the whiteboard |
| 75 min | Tasks 1–6 |
| 30 min | Experiments 2 and 3 |
| 15 min | Discussion |

## Discussion questions

1. You tuned the learning rate by looking at validation accuracy many times. Is the validation score still an honest estimate? (Slightly optimistic: that is what a test set is for.)
2. A dataset of patient records has several rows per patient. How should it be split? (By patient.)
3. Why does dividing by √v̂ make Adam's step size roughly independent of the gradient's scale?
4. Early stopping saved `p` instead of `p.copy()`. What would `restore` do? (Nothing useful: the "saved" arrays are the live ones.)
5. Why check for divergence before overfitting in `diagnose`?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| Momentum/Adam state never changes | Rebound the state array (`v = beta * v + g`) instead of updating in place |
| Adam's first steps tiny | Missing bias correction, or `t` incremented per parameter |
| Early-stopping restore test fails | Saved references, not copies; or restored by rebinding instead of `p[...] =` |
| Split test fails on overlap | Split before shuffling, or overlapping slices |
| `diagnose` calls a nan run "underfitting" | Checked accuracy before checking the loss |

## Checks for understanding

- "Training accuracy 99%, validation 70%. Name it and give two fixes."
- "Training loss: 2.30, 2.30, 2.30, 2.30. Name two possible causes." (Learning rate far too small; or dead ReLUs after a huge first step.)
- "A layer has d_in = 512. What standard deviation does He initialisation use?" (√(2/512) = 0.0625.)
