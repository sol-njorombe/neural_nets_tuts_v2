# Lesson 3 — NumPy and shapes bootcamp

**Read [THEORY.md](THEORY.md) first.** There are no neural networks in this lesson.

## What you build

Short drills, each testing one array skill: reshaping, matrix multiplication
(once by hand, then with `@`), NumPy's broadcasting rule written out yourself,
reductions along an axis, one-hot encoding, and finally the backward rule for
`Y = X @ W` that every later lesson uses.

## Setup

```bash
pytest lesson03_shapes -x
```

**Rule:** no Python loops, except in the three functions whose docstrings allow
them (`matmul_by_loops`, `broadcast_shape`, `sum_of_outer_products`). Those exist
so you see what the fast versions do.

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `as_column`, `as_row` | §2, §6 |
| 2 | `examples_to_arrays`: lesson 1 data to `X (n, d_in)` and `Y (n, 1)` | §1 |
| 3 | `matmul_by_loops`, then `weighted_sums` in one line | §3 |
| 4 | `broadcast_shape`: NumPy's rule, implemented by you | §4 |
| 5 | `column_means`, `normalise_rows`, `one_hot` | §5 |
| 6 | `sum_of_outer_products` with a loop, then with one `@` | §7 |
| 7 | `matmul_backward`: dX and dW for `Y = X @ W` | §8 |

## Experiments

Open a Python prompt in the repo root:

```python
import numpy as np
A = np.arange(6.0).reshape(2, 3)
```

1. Predict the shape of each before running it: `A.T`, `A @ A.T`, `A.T @ A`,
   `A.sum(axis=0)`, `A.sum(axis=1, keepdims=True)`, `A - A.mean(axis=0)`,
   `A - A.mean(axis=1)`. Which one raises an error, and why?
2. Make `y = np.ones(8)` and `a = np.full((8, 1), 0.5)`. Compute `np.mean((a - y) ** 2)`.
   It should be 0.25. What did you get? Fix it with one `reshape`.
3. Time `matmul_by_loops` against `@` on two `(200, 200)` matrices with `%timeit`
   (in IPython) or `time.perf_counter()`.

## The limit

Nothing to fix yet: this lesson is a tool. Lesson 4 uses it to rewrite lesson 2's
neuron for a whole batch of examples at once, with no loops, and derives its
gradient using the table at the end of THEORY §8.

<details>
<summary>Hints (open only after trying)</summary>

- **examples_to_arrays:** `np.array([e.x for e in examples], dtype=np.float64)`.
  For Y, build a flat array of targets and `.reshape(-1, 1)`.
- **matmul_by_loops:** unpack `n, k = A.shape` and `k2, m = B.shape`; check
  `k == k2`; fill `np.zeros((n, m))`.
- **broadcast_shape:** pad both shapes to the same length with `(1,) * extra + shape`,
  then `zip` them.
- **normalise_rows:** `A.sum(axis=1, keepdims=True)`.
- **one_hot:** start from `np.zeros((n, k))`, then assign 1.0 at `[np.arange(n), labels]`.
- **matmul_backward:** write the four shapes down on paper and match inner sizes.
</details>
