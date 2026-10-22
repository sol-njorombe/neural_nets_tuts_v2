# Lesson 1 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Name the parts of a neuron (inputs, weights, bias, weighted sum, activation) and say what each does.
2. Compute a forward pass by hand.
3. Explain why the activation is smooth rather than a hard step.
4. Explain the bias as a threshold, and choose weights by hand for OR and AND.
5. Describe a single neuron's decision as a flat cut through the space of inputs.

## Suggested session (about 1.5 hours)

| Time | Activity |
|---|---|
| 10 min | Mixing-desk analogy; draw the neuron diagram together |
| 10 min | Learner computes THEORY §2's worked example by hand |
| 45 min | Tasks 1–5 |
| 15 min | Experiments, especially 3 and 4 |
| 10 min | Discussion questions |

## Discussion questions

1. An input is 0 for every example in the dataset. What can its weight learn? (Nothing: it is multiplied by zero.)
2. What does the bias let a neuron do that weights alone cannot? (Hint: inputs all zero.)
3. Two neurons both get 100% accuracy on OR3. One outputs 0.99 and 0.01, the other 0.6 and 0.4. Which is better, and why might it matter?
4. Draw the cube of 8 inputs. Can you sketch a flat cut for AND? For XOR?
5. Why does `weighted_sum` raise an error on mismatched lengths instead of using `zip` quietly?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| `OverflowError` in the sigmoid tests | Used `1 / (1 + math.exp(-z))` for all z |
| Reproducibility test fails | Created a new generator inside `random()` instead of using `rng` |
| `type(w) is float` test fails | Stored NumPy scalars; wrap in `float(...)` |
| Bias test fails | Started the sum at 0 instead of `self.b` |
| HAND_OR right but the confidence test fails | Weights too small; scale them up |

## Checks for understanding

- "w = [2, 2], b = −3. Which of the inputs (0,0), (0,1), (1,0), (1,1) give an output above 0.5?" (Only (1,1): sums −3, −1, −1, 1.)
- "Double every weight and the bias. Do the predictions change? Does the confidence?" (Predictions no; confidence yes.)
- "Why does the neuron need a smooth activation if `predict` rounds the output anyway?" (Learning follows slopes; a hard step has none.)
