# Lesson 4 — Theory: one neuron on a whole batch

Same neuron as lesson 2, same maths, same results. What changes is that every
example goes through at once, as one matrix, and the gradient becomes two lines.
This lesson is the bridge between the loop version you understand and the
matrix version every later lesson uses.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Map Loops to Matrix Operations**: Translate each line of Lesson 2's scalar loops into batched NumPy matrix equations.
2. **Track Batched Intermediate Shapes**: Specify the exact dimensions of $Z$, $A$, $dA$, $dZ$, $dW$, and $db$ for arbitrary batch size $n$ and input dimension $d_{\text{in}}$.
3. **Understand Vectorised Backpropagation**: Explain why $X^T @ dZ$ computes "$\delta$ times input, summed over examples" in a single BLAS dot product.
4. **Explain Bias Accumulation**: Explain why the bias gradient $db$ is the sum of $dZ$ across batch rows (`axis=0`).
5. **Analyze Vectorisation vs Expressivity**: Explain why vectorisation produces dramatic speedups (50–100×) but does not change expressivity (the XOR limit remains).

---

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

---

## 2. The forward pass for every example at once

Given feature matrix $X \in \mathbb{R}^{n \times d_{\text{in}}}$, weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times 1}$, and bias $b \in \mathbb{R}^{1}$:

```
X (n, d_in) @ W (d_in, 1) + b (1,)  →  Z (n, 1)       row k = example k's weighted sum
A = σ(Z)                            →  A (n, 1)       row k = example k's output
```

$\sigma$ is applied element by element:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Math | Shape | Code |
|---|---|---|
| $z_k = \sum_i x_{ki} w_i + b$, for every $k$ | `(n, d_in) @ (d_in, 1) + (1,) → (n, 1)` | `Z = X @ W + b` |
| $a_k = \sigma(z_k)$ | `(n, 1)` | `A = sigmoid(Z)` |
| $L = \frac{1}{n} \sum_{k=0}^{n-1} (a_k - y_k)^2$ | `(n, 1)` and `(n, 1) → float` | `float(np.mean((A - Y) ** 2))` |

`Y` must be `(n, 1)` to match `A`. With a flat `(n,)` array for `Y`, lesson 3 §4 showed that
`A - Y` silently broadcasts to shape `(n, n)`: an $n \times n$ table of every prediction minus every target.

---

## 3. The backward pass, from lesson 2 to two lines

Lesson 2 computed, for each example $k$:

$$\delta_k = 2(a_k - y_k) \cdot a_k(1 - a_k)$$

Then averaged over all examples:

$$\frac{\partial L}{\partial w_i} = \frac{1}{n} \sum_{k=0}^{n-1} \delta_k x_{ki}, \qquad \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{k=0}^{n-1} \delta_k$$

Now compute each step for the entire batch at once. By folding the factor $\frac{1}{n}$ directly into $dA$ at the start:

```
dA = 2 * (A - Y) / n             (n, 1)       ∂L/∂A, one entry per example
dZ = dA * A * (1 - A)            (n, 1)       δ for every example (elementwise *)
dW = X.T @ dZ                    (d_in, 1)    Σ_k x_ki δ_k for each input i
db = dZ.sum(axis=0)              (1,)         Σ_k δ_k
```

### Why folding $1/n$ early works
Because the loss $L = \frac{1}{n} \sum_k (a_k - y_k)^2$ is a mean:
$$\frac{\partial L}{\partial a_k} = \frac{2(a_k - y_k)}{n}$$
By incorporating $1/n$ into $dA$, every downstream gradient ($dZ$, $dW$, $db$) automatically becomes the gradient of the dataset mean loss without needing an explicit division at the end.

### Why `X.T @ dZ` computes the weight gradient
Recall from Lesson 3 §7 that $X^T @ dZ$ decomposes into a sum of outer products:

$$(X^T dZ)_{i, 0} = \sum_{k=0}^{n-1} X_{ki} dZ_{k0} = \sum_{k=0}^{n-1} x_{ki} \delta_k = \frac{\partial L}{\partial w_i}$$

The transpose turns rows (examples) into columns, and the matrix multiplication contracts the example dimension, accumulating the gradient across all $n$ examples.

### Why `dZ.sum(axis=0)` computes the bias gradient
Since bias $b$ is broadcast to every row of $Z$, each example provides a path to $L$: $\frac{\partial Z_k}{\partial b} = 1$.
By the multivariable chain rule:
$$\frac{\partial L}{\partial b} = \sum_{k=0}^{n-1} \frac{\partial L}{\partial Z_k} \frac{\partial Z_k}{\partial b} = \sum_{k=0}^{n-1} dZ_k \cdot 1 = \sum_{k=0}^{n-1} dZ_k$$
Summing down the rows (`axis=0`) produces the exact vector of bias gradients.

| Math | Shape | Code |
|---|---|---|
| $\frac{\partial L}{\partial a_k} = \frac{2(a_k - y_k)}{n}$ | `(n, 1)` | `dA = 2 * (A - Y) / n` |
| $\delta_k = \frac{\partial L}{\partial a_k} \cdot a_k(1 - a_k)$ | `(n, 1)` | `dZ = dA * A * (1 - A)` |
| $\frac{\partial L}{\partial w_i} = \sum_k x_{ki} \delta_k$ | `(d_in, 1)`, matching `W` | `dW = X.T @ dZ` |
| $\frac{\partial L}{\partial b} = \sum_k \delta_k$ | `(1,)`, matching `b` | `db = dZ.sum(axis=0)` |

### Worked Example: Zero Neuron on OR3
For the all-zero initialization on OR3 ($n=8$ examples, 1 false, 7 true):
Predictions are all $A_k = \sigma(0) = 0.5$.

$$dA = \frac{2(0.5 - Y)}{8} = \begin{cases} +0.125 & \text{for } y=0 \text{ (row 0)} \\ -0.125 & \text{for } y=1 \text{ (rows 1--7)} \end{cases}$$

$$dZ = dA \cdot (0.5)(1 - 0.5) = \begin{cases} +0.03125 & \text{row 0} \\ -0.03125 & \text{rows 1--7} \end{cases}$$

For any input feature $x_i$, it is 1 on exactly 4 of the positive examples (rows with $y=1$):
$$dW_i = 4 \times (-0.03125) = -0.125$$

For bias $b$:
$$db = (+0.03125) + 7 \times (-0.03125) = -0.1875$$

These match the scalar calculations from Lesson 2 §4 to the last decimal place!

---

## 4. What the forward pass saves

The backward pass requires:
- `X` to compute $dW = X^T @ dZ$
- `A` to compute the activation slope $A * (1 - A)$

The forward pass bundles these into a lightweight `Cache(X=X, A=A)`.
Storing `A` instead of recomputing $\sigma(Z)$ saves significant computation. This caching pattern is identical to Lesson 0's `Trace`, and forms the basis of automatic differentiation in Lesson 8.

---

## 5. Comparison: Loop vs Batched Formulation

| Dimension | Lesson 2 (Scalar Loop) | Lesson 4 (Batched Matrix) |
|---|---|---|
| **Parameters** | `w: list[float]`, `b: float` | `W: FloatArray (d_in, 1)`, `b: FloatArray (1,)` |
| **Loops** | epochs $\times$ examples $\times$ weights | epochs only |
| **Gradient** | Inner loop accumulating $\delta \cdot x$ | Vectorised: `X.T @ dZ` |
| **Runtime (10,000 ex, 100 features)** | $\sim 10^6$ Python bytecode ops / epoch ($\sim 1\text{s}$) | BLAS GEMM / epoch ($\sim 1\text{ms}$, 100× speedup) |
| **Can it solve XOR?** | **No** (linear boundary) | **No** (linear boundary) |

Matrices make the neuron dramatically faster, but cannot alter the geometry of a single hyperplane. To solve non-linearly separable problems like XOR, we need hidden layers. Lesson 5 builds a multilayer perceptron by chaining these exact matrix operations.
