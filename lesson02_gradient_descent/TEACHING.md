# Lesson 2 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Explain why training uses a smooth loss rather than accuracy.
2. Derive ∂L/∂wᵢ = mean of δ·xᵢ for one sigmoid neuron, naming each factor.
3. Explain the minus sign in the update rule and the role of the learning rate.
4. Recognise a learning rate that is too small or too large from a loss history.
5. Explain, with the cube picture, why one neuron cannot compute XOR.

## Suggested session (about 2 hours)

| Time | Activity |
|---|---|
| 10 min | Hot-and-cold game, then the foggy hillside |
| 20 min | Learner works THEORY §4's worked example on paper |
| 50 min | Tasks 1–4 |
| 25 min | Experiments 1 and 4 |
| 15 min | Discussion questions |

## Discussion questions

1. Why square the error instead of taking its absolute value? (Both work; squaring is smooth at zero and punishes big misses more.)
2. Why is every weight's slope "δ times its input"? What does a zero input mean for its weight?
3. The loss history goes 0.25, 0.20, 0.31, 0.18, 0.40… What would you change?
4. The XOR loss gets stuck at 0.25. Is gradient descent broken? How would you convince someone it is not?
5. Wiggling was competitive on 3 inputs. Why does nobody train real networks that way?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| Gradient check off by a factor of n | Summed instead of averaging |
| Bias gradient wrong, weights right | Multiplied the bias term by an input |
| Loss rises during training | Added the gradient instead of subtracting it |
| `history[0]` test fails | Recorded the loss after the step |
| XOR test "fails" because it learned XOR | Not possible with one neuron: check the dataset passed in |

## Checks for understanding

- "δ for an example is −0.3 and its inputs are (1, 0, 1). What are its contributions to ∂L/∂w?" (−0.3, 0, −0.3, before averaging.)
- "A slope is +2 and η = 0.1. What happens to the weight?" (Falls by 0.2.)
- "Name a function of three 0/1 inputs, other than XOR, that one neuron cannot learn." (For example "exactly one input on".)
