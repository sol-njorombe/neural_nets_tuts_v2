# Lesson 3 — NumPy and shapes bootcamp

**Read [THEORY.md](THEORY.md) first.** There are no neural networks in this lesson.

## What you build

Short drills, each testing one array skill: reshaping, matrix multiplication
(once by hand, then with `@`), NumPy's broadcasting rule written out yourself,
reductions along an axis, one-hot encoding, and finally the backward rule for
`Y = X @ W` that every later lesson uses.

## Setup

Run the test suite for this lesson:

```bash
pytest lesson03_shapes -x
```

**Rule for this lesson:** no Python loops, except in the three functions whose
docstrings explicitly permit them (`matmul_by_loops`, `broadcast_shape`,
`sum_of_outer_products`). Those exist so you implement and understand the
low-level mechanics before relying on vectorized NumPy operations.

---

## Tasks

| # | What | Target Shape | Theory Reference |
|---|---|---|---|
| 1 | `as_column`, `as_row` | `(n, 1)`, `(1, n)` | [THEORY §2, §6](THEORY.md#2-shapes) |
| 2 | `examples_to_arrays` | `X (n, d_in)`, `Y (n, 1)` | [THEORY §1](THEORY.md#1-why-arrays) |
| 3 | `matmul_by_loops`, `weighted_sums` | `(n, m)`, `(n, d_out)` | [THEORY §3](THEORY.md#3-matrix-multiplication-) |
| 4 | `broadcast_shape` | Broadcast tuple | [THEORY §4](THEORY.md#4-broadcasting) |
| 5 | `column_means`, `normalise_rows`, `one_hot` | `(d,)`, `(n, d)`, `(n, k)` | [THEORY §5](THEORY.md#5-reductions-axis-and-keepdims) |
| 6 | `sum_of_outer_products` (loop vs fast `@`) | `(d_in, d_out)` | [THEORY §7](THEORY.md#7-matrix-multiply-as-a-sum-over-examples) |
| 7 | `matmul_backward` | `dX (n, d_in)`, `dW (d_in, d_out)` | [THEORY §8](THEORY.md#8-the-backward-rule-for-y--x--w) |

### Task 1: `as_column` and `as_row`
- `as_column(v)`: takes rank-1 array `(n,)` and returns column matrix `(n, 1)`. Use `.reshape(-1, 1)`.
- `as_row(v)`: takes rank-1 array `(n,)` and returns row matrix `(1, n)`. Use `.reshape(1, -1)`.

### Task 2: `examples_to_arrays`
- Converts a list of `Example(x=(...), y=...)` into feature matrix `X` of shape `(n, d_in)` and target column `Y` of shape `(n, 1)` with `dtype=np.float64`.
- **Trap**: do NOT return `Y` with shape `(n,)`. Always return 2D column `(n, 1)` to prevent silent broadcasting bugs.

### Task 3: `matmul_by_loops` and `weighted_sums`
- `matmul_by_loops(A, B)`: checks that inner dimensions match (`A.shape[1] == B.shape[0]`), raising `ValueError` otherwise. Computes $C_{rc} = \sum_j A_{rj} B_{jc}$ using nested loops.
- `weighted_sums(X, W, b)`: computes affine transformation $Z = XW + b$ in a single vectorized line using `@` and broadcasting.

### Task 4: `broadcast_shape`
- Implements NumPy's shape broadcasting logic:
  1. Pad shorter shape with 1s on the left.
  2. For each axis from right to left, sizes must be equal or one must be 1. Raise `ValueError` if incompatible.
  3. The resulting size along that axis is $\max(s_1, s_2)$.

### Task 5: Reductions and Fancy Indexing
- `column_means(A)`: average down the rows, collapsing axis 0 to return shape `(d,)`.
- `normalise_rows(A)`: divide each row by its row sum. Must use `keepdims=True` so the sum has shape `(n, 1)` and broadcasts safely across columns.
- `one_hot(labels, k)`: creates an $(n, k)$ float matrix where entry `[i, labels[i]] = 1.0` using fancy indexing `[np.arange(n), labels]`. No loops!

### Task 6: Matrix Multiply as Sum of Outer Products
- `sum_of_outer_products(X, D)`: computes $\sum_{r=0}^{n-1} \text{outer}(X[r], D[r])$ with an explicit loop over rows.
- `sum_of_outer_products_fast(X, D)`: computes the exact same result using a single matrix multiply: `X.T @ D`.

### Task 7: `matmul_backward`
- Given $dY = \frac{\partial L}{\partial Y}$ for $Y = X @ W$, computes:
  - $dX = dY @ W^T$, shape `(n, d_in)`
  - $dW = X^T @ dY$, shape `(d_in, d_out)`

---

## Experiments

Open a Python REPL or IPython session in the repo root:

```python
import numpy as np
A = np.arange(6.0).reshape(2, 3)
```

1. **Shape Prediction Game**: Predict the shape of each expression before running it:
   - `A.T` $\to$ `(3, 2)`
   - `A @ A.T` $\to$ `(2, 2)`
   - `A.T @ A` $\to$ `(3, 3)`
   - `A.sum(axis=0)` $\to$ `(3,)`
   - `A.sum(axis=1, keepdims=True)` $\to$ `(2, 1)`
   - `A - A.mean(axis=0)` $\to$ `(2, 3)` (centers each column)
   - `A - A.mean(axis=1)` $\to$ **Error!** `(2, 3)` minus `(2,)` aligns as `(2, 3)` vs `(1, 2)`, which fails. To fix, use `A.mean(axis=1, keepdims=True)`.
2. **The Silent Broadcasting Trap**:
   ```python
   y = np.ones(8)          # (8,)
   a = np.full((8, 1), 0.5) # (8, 1)
   diff = a - y             # Shape is (8, 8)!
   print(np.mean(diff ** 2)) # Produces 0.25 by accident here, but gradients get corrupted!
   ```
   Now try with `y = np.array([0, 1, 0, 1, 0, 1, 0, 1])`. Compare `(a - y.reshape(-1, 1)) ** 2` vs `(a - y) ** 2`.
3. **Speed Benchmark**:
   ```python
   import time
   from lesson03_shapes.shapes import matmul_by_loops
   A = np.random.randn(200, 200)
   B = np.random.randn(200, 200)

   t0 = time.perf_counter()
   _ = matmul_by_loops(A, B)
   t_loop = time.perf_counter() - t0

   t0 = time.perf_counter()
   _ = A @ B
   t_blas = time.perf_counter() - t0

   print(f"Loop: {t_loop:.3f}s | BLAS: {t_blas:.6f}s | Speedup: {t_loop/t_blas:.1f}x")
   ```
   Observe speedups between 50× and 500×.

---

## Self-Check Questions

1. **Orientation**: Why do we put examples in rows and each neuron's weights in columns of $W$, rather than the opposite?
   *(Convention: it makes $X @ W$ read left-to-right as data flow. Note that PyTorch `nn.Linear` stores weights transposed as $(d_{\text{out}}, d_{\text{in}})$, which we examine in Lesson 9).*
2. **Dimensions**: What is the practical difference in NumPy between `(n,)` and `(n, 1)`?
   *(A shape of `(n,)` is a 1D array with no row or column orientation; when broadcast against `(n, 1)`, it expands along axis 0 into an $(n, n)$ matrix).*
3. **Bias Accumulation**: Why does the bias gradient $db$ require summing over rows (`axis=0`)?
   *(Because the same bias vector $b$ was added to every single row of the batch during the forward pass; by the multivariable chain rule, all paths from $b$ through every example accumulate).*
4. **Dimension Cancellation**: In $X^T @ dY$, which dimension disappears during matrix multiplication, and what does that algebraic contraction represent?
   *(The batch dimension $n$ contracts away, summing the local parameter gradients across all batch training examples).*
5. **Shape Trap**: When might matching shapes mislead you in deriving backprop rules?
   *(When two dimensions are equal, e.g. square matrices where $n = d_{\text{in}}$, an incorrect order like $dY.T @ X$ might produce the right shape but completely wrong values. Gradient checking catches this).*

---

## Common Mistakes & Troubleshooting

| Symptom | Root Cause | Fix |
|---|---|---|
| Loss is a plausible number, but training does not decrease error | `(n, 1) - (n,)` broadcasted to an `(n, n)` matrix | Keep targets as 2D columns: `y.reshape(-1, 1)` |
| `normalise_rows` passes on rectangular matrix, fails on square | Missing `keepdims=True` in `.sum(axis=1)` | Use `A.sum(axis=1, keepdims=True)` |
| `one_hot` fails with IndexError or TypeError | Using Python `for` loops or incorrect tuple indexing | Use fancy indexing: `P[np.arange(n), labels] = 1.0` |
| `matmul_backward` shapes match, but test values fail | Swapped operands or incorrect transposes (e.g. `dY.T @ X`) | Use $dW = X^T @ dY$ and $dX = dY @ W^T$ |
| Inner dimension mismatch error in `@` | Incompatible shapes passed into matmul | Print array shapes before the operation: `print(A.shape, B.shape)` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`examples_to_arrays`**: `np.array([e.x for e in examples], dtype=np.float64)`.
  For `Y`, build an array of targets and call `.reshape(-1, 1)`.
- **`matmul_by_loops`**: unpack `n, k = A.shape` and `k2, m = B.shape`; check `k == k2`; initialize `C = np.zeros((n, m))`.
- **`broadcast_shape`**: pad both shape tuples on the left to equal length using `(1,) * extra + s`, then loop through `zip(padded_s1, padded_s2)` from right to left.
- **`normalise_rows`**: `A / A.sum(axis=1, keepdims=True)`.
- **`one_hot`**: initialize `P = np.zeros((n, k), dtype=np.float64)`, then set `P[np.arange(n), labels] = 1.0`.
- **`matmul_backward`**: write the shapes down on paper and match inner dimensions:
  $dX = dY @ W^T$ has shape $(n, d_{\text{out}}) \times (d_{\text{out}}, d_{\text{in}}) \to (n, d_{\text{in}})$.
  $dW = X^T @ dY$ has shape $(d_{\text{in}}, n) \times (n, d_{\text{out}}) \to (d_{\text{in}}, d_{\text{out}})$.
</details>
