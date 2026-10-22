# Lesson 3 — Teaching notes

This lesson exists because matrix mechanics (transposes, reshapes, axes) were the
main obstacle in later lessons of the first version of this course. Do not rush it.

## Learning objectives

By the end, the learner can:

1. State the shape of any `@`, `*`, `+`, reduction or reshape result before running it.
2. Explain what `X @ W + b` computes in terms of examples and neurons.
3. Apply the broadcasting rule by hand, and spot the `(n, 1)` versus `(n,)` trap.
4. Choose `axis` and `keepdims` correctly for a row-wise or column-wise operation.
5. Derive `dW = X.T @ dY` and `dX = dY @ W.T` from shapes, and explain the sum over rows.

## Suggested session (about 2 hours)

| Time | Activity |
|---|---|
| 15 min | Shape-prediction game: teacher writes expressions, learner predicts shapes |
| 15 min | Learner computes THEORY §3's worked example by hand |
| 60 min | Tasks 1–7 |
| 20 min | Experiments 1–2 |
| 10 min | Discussion questions |

## Discussion questions

1. Why put examples in rows and each neuron's weights in a column of W, rather than the other way round? (Convention; it makes `X @ W` read left to right like the data flows. PyTorch stores `Linear` weights the other way round, which lesson 9 shows.)
2. What is the difference between `(n,)` and `(n, 1)`? When does it matter?
3. Why does the bias gradient sum over rows?
4. In `X.T @ D`, which dimension disappears, and what does that sum mean for a gradient?
5. Two ways to find `dW` were shown. When might the shape method mislead you? (When two dimensions have equal sizes, so a wrong product still has the right shape. Gradient checks catch that.)

## Common mistakes

| Symptom | Likely cause |
|---|---|
| Loss is a plausible number but training does nothing | `(n, 1) − (n,)` broadcasting to `(n, n)` |
| `normalise_rows` passes on rectangular input, fails on square | Missing `keepdims=True` |
| `one_hot` uses a loop | Not yet comfortable with fancy indexing: revisit THEORY §5 |
| `matmul_backward` shapes right, values wrong | Swapped operands, e.g. `dY.T @ X` then transposed |

## Checks for understanding

- "X is (32, 64), W is (64, 10). Shape of X @ W? Of X.T? Of (X @ W).sum(axis=0)?" ((32, 10), (64, 32), (10,).)
- "Z is (32, 10) and b is (10,). What does Z + b do?" (Adds b to every row.)
- "dY is (32, 10), X is (32, 64). Shape of dW and the expression for it?" ((64, 10); X.T @ dY.)
