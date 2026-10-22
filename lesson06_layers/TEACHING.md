# Lesson 6 — Teaching notes

The longest lesson. Consider two sessions: tasks 1–3, then 4–5 and experiments.

## Learning objectives

By the end, the learner can:

1. State the layer contract and explain why it makes layers composable.
2. Implement forward and backward for Dense, ReLU and Sigmoid from the lesson 3 table.
3. Explain the vanishing gradient problem and why ReLU helps.
4. Explain softmax, cross-entropy, and why their combined gradient is P − one-hot.
5. Explain minibatch SGD as a compromise between full-batch and one-at-a-time.

## Suggested sessions (2 × 1.5 hours)

| Session | Activity |
|---|---|
| 1 | Bricks and the contract (15 min); THEORY §4 worked example (10 min); tasks 1–3 (65 min) |
| 2 | Tasks 4–5 (50 min); experiments 1–3 (30 min); discussion (10 min) |

## Discussion questions

1. Why does a layer's `backward` return blame for its input, not for its parameters?
2. Why does the Sigmoid layer save its output while ReLU saves a mask? What does each layer need to remember, in general?
3. Cross-entropy punishes a confident mistake far more than an unsure one. Why is that useful for training? Could it be harmful? (Mislabelled data gets huge loss.)
4. With batch size 1437, each epoch is one step. With batch size 1, 1437 steps. Which learns more per epoch? Which per second?
5. What would you need to add a convolution layer? (A forward and a backward obeying the contract. Nothing else changes.)

## Common mistakes

| Symptom | Likely cause |
|---|---|
| Training does nothing, no errors | `self.dW = ...` or `p = p - lr * g` instead of in-place |
| `nan` in softmax | Did not subtract the row max |
| Softmax rows don't sum to 1 | Missing `keepdims=True` |
| Deep-stack gradient test fails, single layers pass | `Sequential.backward` not reversed, or not passing the returned blame along |
| Accuracy stuck near 10% | Labels and rows misaligned in `minibatches` (indexed X and y with different orders) |

## Checks for understanding

- "A batch of 32 through `Dense(64, 32)`, `ReLU`, `Dense(32, 10)`. Shapes of each output and of each dW?" ((32, 32), (32, 32), (32, 10); (64, 32), (32, 10).)
- "Probabilities [0.7, 0.2, 0.1], correct class 1. Loss? Gradient on the logits?" (−ln 0.2 ≈ 1.61; [0.7, −0.8, 0.1], before dividing by n.)
