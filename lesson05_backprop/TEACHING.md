# Lesson 5 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Explain how a hidden layer lets a network compute XOR, using the hand-built example.
2. Write the forward pass for two layers and state every shape.
3. Derive the backward pass using lesson 4's three lines plus `dH = dZ2 @ W2.T`.
4. State the general per-layer backprop pattern in four lines.
5. Explain local minima and symmetry as properties of training, not bugs.

## Suggested session (about 2.5 hours)

| Time | Activity |
|---|---|
| 15 min | Specialists analogy; learner fills in THEORY §2's table |
| 20 min | Task 1 (hand-built XOR) |
| 15 min | Learner works THEORY §5's example on paper |
| 60 min | Tasks 2–4 |
| 30 min | Experiments 1–3 |
| 10 min | Discussion |

## Discussion questions

1. The hand-built network uses three hidden neurons. Can you do XOR3 with two? (The experiment says training sometimes succeeds with two, so yes; ask the learner to find out what those two neurons compute.)
2. Why does the blame passed to the hidden layer get multiplied by W2?
3. Why does every hidden neuron end up identical if all weights start equal?
4. After training, are the hidden neurons the same as the hand-built ones? Does it matter?
5. What would you need to change to add a second hidden layer? (Every line of forward and backward: the motivation for lesson 6.)

## Common mistakes

| Symptom | Likely cause |
|---|---|
| dW2 correct, dW1 wrong | Used `A * (1 - A)` in the hidden layer instead of `H * (1 - H)` |
| `ValueError` in `dZ2 @ W2` | Forgot the transpose: `W2.T` |
| XOR test fails at 75% | A real local minimum is possible with other settings; check d_hidden = 4, lr = 2, epochs = 5000 |
| `HAND_XOR` outputs 0.5-ish | Weights too small; the sigmoids are not decisive |

## Checks for understanding

- "d_in = 3, d_hidden = 5, n = 8. Shapes of H, dZ2, dH, dW1?" ((8, 5), (8, 1), (8, 5), (3, 5).)
- "A hidden neuron's outgoing weight W2[j] is 0. What is its gradient dW1[:, j]?" (Zero: it gets no blame.)
- "State the four-line backward pattern for a generic layer."
