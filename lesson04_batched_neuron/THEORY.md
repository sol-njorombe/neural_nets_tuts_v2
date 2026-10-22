# Lesson 4 — Theory: one neuron on a whole batch

Same neuron as lesson 2, same maths, same results. What changes is that every
example goes through at once, as one matrix, and the gradient becomes two lines.
This lesson is the bridge between the loop version you understand and the
matrix version every later lesson uses.

## 1. A neuron is a layer with one output

Put the neuron's weights in a single column:

```
lesson 2:   w = [w₀, w₁, w₂]        b = 0.3          (Python list and float)
lesson 4:   W = ┌ w₀ ┐  (3, 1)      b = [0.3]  (1,)  (NumPy arrays)
                │ w₁ │
                └ w₂ ┘
```

Why a column, not a flat array? Because lesson 5 has several neurons side by
side, one column each: `W (d_in, d_out)`. A single neuron is the special case
`d_out = 1`. Keeping it two-dimensional now means no code changes later.

## 2. The forward pass for every example at once

```
X (n, d_in) @ W (d_in, 1) + b (1,)  →  Z (n, 1)       row k = example k's weighted sum
A = σ(Z)                            →  A (n, 1)       row k = example k's output
```

`σ` is applied element by element, so it works on any shape.

| Math | Shape | Code |
|---|---|---|
| zₖ = Σᵢ xₖᵢwᵢ + b, for every k | (n, d_in) @ (d_in, 1) + (1,) → (n, 1) | `Z = X @ W + b` |
| aₖ = σ(zₖ) | (n, 1) | `A = sigmoid(Z)` |
| L = (1/n) Σₖ (aₖ − yₖ)² | (n, 1) and (n, 1) → one number | `float(np.mean((A - Y) ** 2))` |

`Y` must be `(n, 1)` to match `A`. With a flat `(n,)` Y, lesson 3 §4 showed that
`A − Y` silently becomes `(n, n)`.

## 3. The backward pass, from lesson 2 to two lines

Lesson 2 computed, for each example k:

```
δₖ = 2(aₖ − yₖ) · aₖ(1 − aₖ)          then   ∂L/∂wᵢ = (1/n) Σₖ δₖ xₖᵢ      ∂L/∂b = (1/n) Σₖ δₖ
```

Do each step for every example at once. Fold the `1/n` in at the start, since it
comes from L being a mean:

```
dA = 2(A − Y) / n                (n, 1)       ∂L/∂A, one entry per example
dZ = dA · A(1 − A)               (n, 1)       δ for every example (elementwise *)
dW = X.T @ dZ                    (d_in, 1)    Σₖ xₖᵢ δₖ  for each i
db = dZ.sum(axis=0)              (1,)         Σₖ δₖ
```

The last two lines are the table from lesson 3 §8. `X.T @ dZ` is "each input
times its example's δ, summed over examples" (lesson 3 §7), exactly lesson 2's
inner loop. The bias reached every example's z with slope 1, so its gradient is
the sum of the δs.

| Math | Shape | Code |
|---|---|---|
| ∂L/∂aₖ = 2(aₖ − yₖ)/n | (n, 1) | `dA = 2 * (A - Y) / n` |
| δₖ = ∂L/∂aₖ · aₖ(1 − aₖ) | (n, 1) | `dZ = dA * A * (1 - A)` |
| ∂L/∂wᵢ = Σₖ xₖᵢ δₖ | (d_in, 1), like W | `dW = X.T @ dZ` |
| ∂L/∂b = Σₖ δₖ | (1,), like b | `db = dZ.sum(axis=0)` |

**Worked example.** The zero neuron on OR3 from lesson 2 §4. `A` is all 0.5 and
`n = 8`:

```
dA = 2(0.5 − Y)/8  →  +0.125 for the y=0 row, −0.125 for the seven y=1 rows
dZ = dA · 0.25     →  +0.03125, and −0.03125 seven times
dW = X.T @ dZ      →  each input is 1 on four rows, all y=1:  4 × (−0.03125) = −0.125
db = sum of dZ     →  0.03125 − 7 × 0.03125 = −0.1875
```

The same numbers as lesson 2, as they must be.

## 4. What the forward pass saves

The backward pass needs `X` (for `dW`) and `A` (for the sigmoid slope). The
forward pass returns them in a `Cache`, so nothing is recomputed. This is lesson
0's `Trace` again, and lesson 6 turns it into the layer's own memory.

## 5. What changed and what did not

| | Lesson 2 | Lesson 4 |
|---|---|---|
| Parameters | `list[float]`, `float` | `W (d_in, 1)`, `b (1,)` |
| Loops | epochs × examples × weights | epochs only |
| Gradient | a loop accumulating δ·x | `X.T @ dZ` |
| Speed on 10,000 × 100 | about a million Python steps per epoch | a few milliseconds per epoch |
| Can learn XOR? | No | **No** |

Matrices made the neuron fast, not more capable. One flat cut is still one flat
cut. Lesson 5 adds the hidden layer, and because `W` is already a matrix it
needs no new ideas about shapes, just a second application of the same rules.
