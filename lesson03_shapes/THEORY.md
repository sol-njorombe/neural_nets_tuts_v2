# Lesson 3 — Theory: NumPy and shapes

No neural networks in this lesson. It is a bootcamp in the one skill that makes
every later lesson readable: translating math into array operations, and
knowing the shape of every array at every line.

The payoff in §8 is a rule you will use in every remaining lesson:

```
Y = X @ W     →     dX = dY @ W.T     and     dW = X.T @ dY
```

By the end you will be able to derive it rather than memorise it.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Predict Array Shapes**: State the shape of any `@`, `*`, `+`, reduction (`sum`, `mean`, `max`), transpose, or reshape operation before running it.
2. **Interpret Affine Transforms**: Explain what `X @ W + b` computes in terms of batch examples and individual neuron weights.
3. **Execute Broadcasting Rules**: Apply NumPy's 3-step broadcasting algorithm by hand and debug the silent `(n, 1)` vs `(n,)` outer product bug.
4. **Master Dimensional Reductions**: Select the correct `axis` and explain why `keepdims=True` is required when normalizing rows or columns.
5. **Derive Matrix Backpropagation**: Prove `dW = X.T @ dY` and `dX = dY @ W.T` both through dimension alignment and through the multivariable chain rule over batch rows.

---

## 1. Why arrays

Lesson 2's `gradients` was a loop over examples with a loop over weights inside.
NumPy replaces those loops with operations on whole arrays at once, executed in
compiled C rather than in the Python interpreter. It is commonly 50–100× faster
for this kind of code, and the code ends up closer to the math.

A dataset becomes two arrays. **Rows are always examples** ([NOTATION.md](../NOTATION.md)):

```
                 x₀  x₁  x₂                y
              ┌             ┐           ┌   ┐
  example 0   │ 0   0   0   │           │ 0 │
  example 1   │ 0   0   1   │           │ 1 │
     ...      │     ...     │           │...│
  example 7   │ 1   1   1   │           │ 1 │
              └             ┘           └   ┘
                X (8, 3)                 Y (8, 1)
```

In matrix notation:
- $X \in \mathbb{R}^{n \times d_{\text{in}}}$: row $r$ is the input vector $x^{(r)} \in \mathbb{R}^{d_{\text{in}}}$.
- $Y \in \mathbb{R}^{n \times d_{\text{out}}}$ (or $\mathbb{R}^{n \times 1}$ for single-output binary tasks): row $r$ is the target $y^{(r)}$.

---

## 2. Shapes

Every array has a `.shape`, a tuple of sizes, one per dimension.

| Expression | Shape | Description | Geometry / Memory |
|---|---|---|---|
| `np.array(3.0)` | `()` | 0D scalar | a single number |
| `np.array([1., 2., 3.])` | `(3,)` | 1D rank-1 vector | a flat list of 3 (no concept of row vs col) |
| `np.array([[1.], [2.], [3.]])` | `(3, 1)` | 2D column vector | 3 rows, 1 column |
| `np.array([[1., 2., 3.]])` | `(1, 3)` | 2D row vector | 1 row, 3 columns |
| `np.zeros((8, 3))` | `(8, 3)` | 2D matrix | 8 rows, 3 columns |

**`(3,)`, `(3, 1)` and `(1, 3)` are three different things.** The difference
causes most shape bugs (§4).

Habits that prevent nearly all of them:
- write the shape as a comment on every line that creates an array;
- when confused, `print(A.shape)` before anything else.

---

## 3. Matrix multiplication: `@`

`A @ B` needs the **inner sizes to match**, and the result takes the outer sizes:

```
A (n, k)  @  B (k, m)   →   C (n, m)
      └──────┘ must match
```

Each entry is a row of `A` dotted with a column of `B`:

$$C[r, c] = \sum_{j=0}^{k-1} A[r, j] \cdot B[j, c]$$

**Worked example.**

```
A (3, 2)          B (2, 3)                C = A @ B   (3, 3)
┌      ┐        ┌          ┐            ┌          ┐
│ 1  2 │        │ 1  0   2 │            │ 1  2  0  │   row 0: [1·1+2·0, 1·0+2·1, 1·2+2·(−1)]
│ 3  4 │   @    │ 0  1  −1 │     =      │ 3  4  2  │   row 1: [3·1+4·0, 3·0+4·1, 3·2+4·(−1)]
│ 5  6 │        └          ┘            │ 5  6  4  │   row 2: [5·1+6·0, 5·0+6·1, 5·2+6·(−1)]
└      ┘                                └          ┘
```

**Why this is exactly a layer of neurons.** Put each neuron's weights in a
column of `W`:

```
X (n, d_in)  @  W (d_in, d_out)  +  b (d_out,)   →   Z (n, d_out)

Z[r, j] = Σᵢ X[r, i] · W[i, j] + b[j]  =  the weighted sum of example r through neuron j
```

One `@` replaces three loops: over examples, over neurons, and over weights.

| Math | Shape | Code |
|---|---|---|
| $z_{rj} = \sum_i x_{ri} w_{ij} + b_j$ | `(n, d_in) @ (d_in, d_out) + (d_out,) → (n, d_out)` | `X @ W + b` |

`*` is different: it multiplies **element by element** and needs equal (or
broadcastable) shapes. Mixing up `*` and `@` usually raises a shape error, which
is the good outcome.

---

## 4. Broadcasting

In `X @ W + b`, `b` has shape `(d_out,)` but `X @ W` has shape `(n, d_out)`.
NumPy **broadcasts**: it acts as if `b` were copied onto every row without allocating redundant memory. The rule:

1. Line the shapes up **from the right**. Pad the shorter one on the left with 1s.
2. In each position the sizes must be equal, or one must be 1.
3. A size of 1 is stretched to match. The result takes the larger size.

| a | b | Padded comparison | Result | What happens |
|---|---|---|---|---|
| `(8, 4)` | `(4,)` | `(8, 4)` vs `(1, 4)` | `(8, 4)` | `b` is added to every row |
| `(8, 4)` | `(8, 1)` | `(8, 4)` vs `(8, 1)` | `(8, 4)` | column vector is added across all columns |
| `(8, 4)` | `(8,)` | `(8, 4)` vs `(1, 8)` | **error** | 4 vs 8 cannot match |
| `(8, 1)` | `(8,)` | `(8, 1)` vs `(1, 8)` | **`(8, 8)`** | **Silent bug**: outer product table |

The last row is the classic neural network bug. If predictions `A` are a column `(8, 1)` and targets
`y` are flat `(8,)`:
```
A (8, 1) - y (8,)  →  padded as (8, 1) - (1, 8)  →  yields (8, 8) matrix!
```
Instead of subtracting each target from its corresponding prediction, it computes an $8 \times 8$ matrix of every prediction minus every target. Then `np.mean((A - y) ** 2)` computes the mean over 64 numbers, producing a seemingly reasonable float that does not train the model.

> [!IMPORTANT]
> Always enforce column vectors for 1D targets in batches: use `y.reshape(-1, 1)` so shape is `(n, 1)`.

---

## 5. Reductions: `axis` and `keepdims`

`A.sum()`, `A.mean()` and `A.max()` collapse an array. `axis` specifies which
dimension disappears:

```
A (n, d)
A.sum(axis=0)  →  (d,)    collapse rows: one total per column ("down the columns")
A.sum(axis=1)  →  (n,)    collapse columns: one total per row ("across the rows")
```

`keepdims=True` keeps the collapsed dimension with size 1, so the result
broadcasts safely back against the original:

```
A.sum(axis=1, keepdims=True)  →  (n, 1)

A / A.sum(axis=1, keepdims=True)     (n, d) / (n, 1)  →  each row divided by its own sum ✓
A / A.sum(axis=1)                    (n, d) / (n,)    →  error (if n ≠ d), OR silent bug:
                                                          on a square matrix (n == d),
                                                          each COLUMN is divided by a row sum! ✗
```

**Fancy indexing** selects one entry per row: `P[np.arange(n), labels]` extracts
`P[0, labels[0]]`, `P[1, labels[1]]`, ..., giving shape `(n,)`. You will use it for
one-hot encoding here and for the categorical cross-entropy loss in lesson 6.

---

## 6. Reshape and transpose

- `v.reshape(-1, 1)` turns `(n,)` into a column `(n, 1)`. `-1` means "infer whatever
  size preserves total elements ($n$)".
- `A.T` swaps rows and columns: `(n, d)` becomes `(d, n)`. Entry `[i, j]` moves
  to `[j, i]`. In NumPy, this returns a strided view: **no data is copied**.

---

## 7. Matrix multiply as a sum over examples

`X.T @ D`, with `X (n, d_in)` and `D (n, d_out)`, has shape `(d_in, d_out)`.
Written out, it adds up one outer product matrix per example:

$$X^T D = \sum_{r=0}^{n-1} X[r, :]^T D[r, :]$$

where $\text{outer}(u, v)[i, j] = u[i] \cdot v[j]$ is a $(d_{\text{in}}, d_{\text{out}})$ table.

```
       X.T (d_in, n)             D (n, d_out)               X.T @ D (d_in, d_out)
   ┌                      ┐   ┌                      ┐     ┌                      ┐
   │ x₀⁽⁰⁾  x₀⁽¹⁾ ... x₀⁽ⁿ⁻¹⁾│   │ d₀⁽⁰⁾  d₁⁽⁰⁾ ...     │     │                      │
   │ x₁⁽⁰⁾  x₁⁽¹⁾ ... x₁⁽ⁿ⁻¹⁾│ @ │ d₀⁽¹⁾  d₁⁽¹⁾ ...     │  =  │   Σ_r x_i⁽ʳ⁾ d_j⁽ʳ⁾  │
   │ ...                  │   │ ...                  │     │                      │
   └                      ┘   └                      ┘     └                      ┘
```

So `X.T @ D` means: **for every example $r$, multiply each of its inputs by each of
its $D$ values, and add the results over all $n$ examples.**

Recall lesson 2's weight gradient: "$\delta$ times input, summed over examples".
That is why the batched weight gradient in lesson 4 is written in exactly one line:
`dW = (1 / n) * (X.T @ delta)`.

---

## 8. The backward rule for `Y = X @ W`

Suppose a scalar loss $L$ depends on $Y = X @ W$, and you already know $dY = \frac{\partial L}{\partial Y} \in \mathbb{R}^{n \times d_{\text{out}}}$. You need $dW$ and $dX$.

### Method 1: Shape alignment heuristic

A gradient always has the exact shape of the variable it differentiates ([NOTATION.md](../NOTATION.md)):

```
dW must have shape (d_in, d_out).   Available: X (n, d_in) and dY (n, d_out).
Only one matrix product works:      X.T @ dY       → (d_in, n) @ (n, d_out) = (d_in, d_out)  ✓

dX must have shape (n, d_in).       Available: dY (n, d_out) and W (d_in, d_out).
Only one matrix product works:      dY @ W.T       → (n, d_out) @ (d_out, d_in) = (n, d_in)  ✓
```

### Method 2: Multivariable chain rule derivation

Let's derive $dW_{jc}$ entry by entry. $W_{jc}$ is the weight connecting input feature $j$ to output neuron $c$. In the forward pass:

$$Y_{rc} = \sum_{k=0}^{d_{\text{in}}-1} X_{rk} W_{kc}$$

Notice that $W_{jc}$ only affects column $c$ of $Y$, but it affects $Y_{rc}$ for **every example $r \in \{0, \dots, n-1\}$**. By the multivariable chain rule (Lesson 0 §6):

$$\frac{\partial L}{\partial W_{jc}} = \sum_{r=0}^{n-1} \frac{\partial L}{\partial Y_{rc}} \frac{\partial Y_{rc}}{\partial W_{jc}}$$

Since $\frac{\partial Y_{rc}}{\partial W_{jc}} = X_{rj}$:

$$\frac{\partial L}{\partial W_{jc}} = \sum_{r=0}^{n-1} X_{rj} dY_{rc} = \sum_{r=0}^{n-1} X^T_{jr} dY_{rc} = (X^T @ dY)_{jc}$$

Similarly, for $X_{rj}$: the input $X_{rj}$ of example $r$ feeds forward into all output neurons $c \in \{0, \dots, d_{\text{out}}-1\}$ for that same example:

$$\frac{\partial L}{\partial X_{rj}} = \sum_{c=0}^{d_{\text{out}}-1} \frac{\partial L}{\partial Y_{rc}} \frac{\partial Y_{rc}}{\partial X_{rj}} = \sum_{c=0}^{d_{\text{out}}-1} dY_{rc} W_{jc} = \sum_{c=0}^{d_{\text{out}}-1} dY_{rc} W^T_{cj} = (dY @ W^T)_{rj}$$

Both methods agree completely!

### Bias gradient: $db$

When $Y = X @ W + b$, where $b \in \mathbb{R}^{d_{\text{out}}}$:
Bias $b_c$ is added to every row $r$ of $Y[:, c]$. By the chain rule:

$$\frac{\partial L}{\partial b_c} = \sum_{r=0}^{n-1} \frac{\partial L}{\partial Y_{rc}} \frac{\partial Y_{rc}}{\partial b_c} = \sum_{r=0}^{n-1} dY_{rc} \cdot 1 = \sum_{r=0}^{n-1} dY_{rc}$$

In NumPy, summing over rows corresponds to `axis=0`:
$$db = dY.\text{sum}(\text{axis}=0)$$

---

## 9. Core Operations Summary Table

| Operation | Forward Math | NumPy Code | Output Shape | Backward Gradient |
|---|---|---|---|---|
| **Linear layer** | $Y = XW + b$ | `X @ W + b` | `(n, d_out)` | `dW = X.T @ dY`<br>`dX = dY @ W.T`<br>`db = dY.sum(axis=0)` |
| **Row normalize** | $P_{rc} = \frac{A_{rc}}{\sum_k A_{rk}}$ | `A / A.sum(axis=1, keepdims=True)` | `(n, d)` | Used in Softmax |
| **Column mean** | $\mu_c = \frac{1}{n} \sum_r A_{rc}$ | `A.mean(axis=0)` | `(d,)` | Used in Batch Normalization |
| **One-hot** | $Y_{rc} = \mathbb{I}[y_r = c]$ | `P[np.arange(n), labels] = 1.0` | `(n, k)` | Target encoding |
| **Reshape col** | $v \in \mathbb{R}^n \to \mathbb{R}^{n \times 1}$ | `v.reshape(-1, 1)` | `(n, 1)` | Essential for binary targets |
