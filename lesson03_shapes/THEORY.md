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

## 1. Why arrays

Lesson 2's `gradients` was a loop over examples with a loop over weights inside.
NumPy replaces those loops with operations on whole arrays at once, executed in
compiled C rather than in the Python interpreter. It is commonly 50–100× faster
for this kind of code, and the code ends up closer to the math.

A dataset becomes two arrays. **Rows are always examples** (NOTATION.md):

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

## 2. Shapes

Every array has a `.shape`, a tuple of sizes, one per dimension.

| Expression | Shape | Picture |
|---|---|---|
| `np.array(3.0)` | `()` | a single number |
| `np.array([1., 2., 3.])` | `(3,)` | a flat list of 3 |
| `np.array([[1.], [2.], [3.]])` | `(3, 1)` | a column: 3 rows, 1 column |
| `np.array([[1., 2., 3.]])` | `(1, 3)` | a row: 1 row, 3 columns |
| `np.zeros((8, 3))` | `(8, 3)` | 8 rows, 3 columns |

**`(3,)`, `(3, 1)` and `(1, 3)` are three different things.** The difference
causes most shape bugs (§4).

Habits that prevent nearly all of them:
- write the shape as a comment on every line that creates an array;
- when confused, `print(A.shape)` before anything else.

## 3. Matrix multiplication: `@`

`A @ B` needs the **inner sizes to match**, and the result takes the outer sizes:

```
A (n, k)  @  B (k, m)   →   C (n, m)
      └──────┘ must match
```

Each entry is a row of `A` dotted with a column of `B`:

```
C[r, c] = Σⱼ A[r, j] · B[j, c]
```

**Worked example.**

```
A (3, 2)          B (2, 3)                C = A @ B   (3, 3)
┌      ┐        ┌          ┐            ┌          ┐
│ 1  2 │        │ 1  0   2 │            │ 1  2  0  │   row 0: [1·1+2·0, 1·0+2·1, 1·2+2·(−1)]
│ 3  4 │   @    │ 0  1  −1 │     =      │ 3  4  2  │
│ 5  6 │        └          ┘            │ 5  6  4  │
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
| zᵣⱼ = Σᵢ xᵣᵢ wᵢⱼ + bⱼ | (n, d_in) @ (d_in, d_out) + (d_out,) → (n, d_out) | `X @ W + b` |

`*` is different: it multiplies **element by element** and needs equal (or
broadcastable) shapes. Mixing up `*` and `@` usually raises a shape error, which
is the good outcome.

## 4. Broadcasting

In `X @ W + b`, `b` has shape `(d_out,)` but `X @ W` has shape `(n, d_out)`.
NumPy **broadcasts**: it acts as if `b` were copied onto every row. The rule:

1. Line the shapes up **from the right**. Pad the shorter one on the left with 1s.
2. In each position the sizes must be equal, or one must be 1.
3. A size of 1 is stretched to match. The result takes the larger size.

| a | b | Padded | Result |
|---|---|---|---|
| (8, 4) | (4,) | (8, 4) vs (1, 4) | (8, 4) — add b to every row |
| (8, 4) | (8, 1) | (8, 4) vs (8, 1) | (8, 4) — add a column to every column |
| (8, 4) | (8,) | (8, 4) vs (1, 8) | **error**: 4 vs 8 |
| (8, 1) | (8,) | (8, 1) vs (1, 8) | **(8, 8)** — no error, wrong answer |

The last row is the classic bug. If outputs `A` are a column `(8, 1)` and targets
`y` are flat `(8,)`, then `A − y` silently becomes an 8×8 table of every output
minus every target, and the mean of that is a meaningless number. **Keep single
targets as a column, (n, 1), always.**

## 5. Reductions: `axis` and `keepdims`

`A.sum()`, `A.mean()` and `A.max()` collapse an array. `axis` says which
dimension disappears:

```
A (n, d)
A.sum(axis=0)  →  (d,)    collapse the rows: one total per column ("down")
A.sum(axis=1)  →  (n,)    collapse the columns: one total per row ("across")
```

`keepdims=True` keeps the collapsed dimension with size 1, so the result
broadcasts back against the original:

```
A.sum(axis=1, keepdims=True)  →  (n, 1)

A / A.sum(axis=1, keepdims=True)     (n, d) / (n, 1)  →  each row divided by its own sum ✓
A / A.sum(axis=1)                    (n, d) / (n,)    →  error, or on a square matrix,
                                                         each COLUMN divided by a row sum ✗
```

**Fancy indexing** picks one entry per row: `P[np.arange(n), labels]` takes
`P[0, labels[0]]`, `P[1, labels[1]]`, …, shape `(n,)`. You will use it for
one-hot encoding here and for the cross-entropy loss in lesson 6.

## 6. Reshape and transpose

- `v.reshape(-1, 1)` turns `(n,)` into a column `(n, 1)`. `-1` means "whatever
  size makes the total work".
- `A.T` swaps rows and columns: `(n, d)` becomes `(d, n)`. Entry `[i, j]` moves
  to `[j, i]`. Nothing is copied.

## 7. Matrix multiply as a sum over examples

`X.T @ D`, with `X (n, d_in)` and `D (n, d_out)`, has shape `(d_in, d_out)`.
Written out, it adds up one small table per example:

```
X.T @ D  =  Σ over rows r of  outer(X[r], D[r])

outer(u, v)[i, j] = u[i] · v[j]          a (len u, len v) table
```

So `X.T @ D` means: **for every example, multiply each of its inputs by each of
its D values, and add the results over all examples.** Lesson 2's weight
gradient was exactly this: "δ times the input, summed over the examples". That
is why the weight gradient in lesson 4 is one line: `X.T @ delta`.

## 8. The backward rule for `Y = X @ W`

Suppose a loss L depends on `Y = X @ W`, and you already know `dY = ∂L/∂Y`,
shape `(n, d_out)`. You need `dW` and `dX`.

**Method 1: shapes.** A gradient always has the shape of the thing it belongs
to (NOTATION.md). Combine `dY` with the other operand so the shape comes out
right:

```
dW must be (d_in, d_out).   Have X (n, d_in) and dY (n, d_out).
Only one product works:     X.T @ dY        (d_in, n) @ (n, d_out)  ✓

dX must be (n, d_in).       Have dY (n, d_out) and W (d_in, d_out).
Only one product works:     dY @ W.T        (n, d_out) @ (d_out, d_in)  ✓
```

**Method 2: the chain rule, one entry at a time.** `W[j, c]` appears in
`Y[r, c] = Σⱼ X[r, j] · W[j, c]` for every row r, with local slope `X[r, j]`.
That is n paths from `W[j, c]` to L, and paths add (lesson 0 §6):

```
∂L/∂W[j, c] = Σᵣ  ∂L/∂Y[r, c] · X[r, j]  =  Σᵣ X.T[j, r] · dY[r, c]  =  (X.T @ dY)[j, c]  ✓
```

Both methods agree, and the first is much faster to use. Intuitively, the
transpose appears because the backward pass sends blame along the same wires as
the forward pass, but in the opposite direction.

| Forward | Backward | Shapes |
|---|---|---|
| Y = X @ W | dW = X.T @ dY | (d_in, n) @ (n, d_out) → (d_in, d_out) |
| | dX = dY @ W.T | (n, d_out) @ (d_out, d_in) → (n, d_in) |
| Y = … + b | db = dY.sum(axis=0) | (n, d_out) → (d_out,): b reached every row, so add over rows |

That table is the whole of backpropagation through a dense layer. Lessons 4–6
apply it again and again.
