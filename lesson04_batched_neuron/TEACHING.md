# Lesson 4 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Map each line of lesson 2's loop gradient to a line of matrix code.
2. Give the shape of every intermediate (Z, A, dA, dZ, dW, db) for any n and d_in.
3. Explain why `X.T @ dZ` computes "δ times input, summed over examples".
4. Explain why the bias gradient is a sum over rows.
5. Explain why vectorising made training faster but not more capable.

## Suggested session (about 1.5 hours)

| Time | Activity |
|---|---|
| 15 min | Side by side on a whiteboard: lesson 2's loop and THEORY §3's four lines |
| 15 min | Learner works THEORY §3's example by hand |
| 40 min | Tasks 1–3 |
| 15 min | Experiments 2 and 3 |
| 5 min | Discussion |

## Discussion questions

1. Where did lesson 2's loop over examples go? Where did the loop over weights go?
2. Why fold the `1/n` into `dA` rather than dividing `dW` and `db` at the end? (Either works; folding it in early means every later gradient is automatically "of the mean".)
3. Experiment 3 produced no error. Why is a silent shape bug worse than a crash?
4. Why store a single neuron's weights as `(d_in, 1)` rather than `(d_in,)`?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| `db` shape `(1, 1)` | `keepdims=True` on the bias sum |
| Gradient off by a factor of n | Missing or doubled `/ n` |
| Loss stuck near 0.25 on OR3 | Y passed as `(n,)`: check `Y.shape` |
| `ValueError: matmul` in backward | `X @ dZ` instead of `X.T @ dZ` |

## Checks for understanding

- "n = 100, d_in = 5. Shapes of Z, dZ, dW, db?" ((100, 1), (100, 1), (5, 1), (1,).)
- "If one input column of X is all zeros, what is its row of dW?" (Zero.)
