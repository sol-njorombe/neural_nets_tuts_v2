# Lesson 4 — One neuron on a whole batch

**Read [THEORY.md](THEORY.md) first.**

## What you build

Lesson 2's neuron again, but every example goes through at once as a matrix. The
forward pass is two lines, the gradient is four, and there are no loops except
over epochs. The tests check your gradient against both the numerical wiggle method
and your Lesson 2 loop version.

## Setup

Run the test suite and demonstration script:

```bash
pytest lesson04_batched_neuron -x
python lesson04_batched_neuron/batched.py
```

## The types

| Type | Fields | Shapes | Role |
|---|---|---|---|
| `NeuronParams` | `W`, `b` | `(d_in, 1)`, `(1,)` | Model weights and bias |
| `Cache` | `X`, `A` | `(n, d_in)`, `(n, 1)` | Stored intermediate activations for backward pass |
| `Grads` | `dW`, `db` | `(d_in, 1)`, `(1,)` | Gradients with exact parameter shapes |

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `forward` and `mse` | `(params, X) -> (A, Cache)`, `(A, Y) -> float` | [THEORY §2](THEORY.md#2-the-forward-pass-for-every-example-at-once) |
| 2 | `backward` | `(params, Y, cache) -> Grads` | [THEORY §3](THEORY.md#3-the-backward-pass-from-lesson-2-to-two-lines) |
| 3 | `train` | `(params, X, Y, lr, epochs) -> list[float]` | [THEORY §3, §5](THEORY.md#5-comparison-loop-vs-batched-formulation) |

### Task 1: `forward` and `mse`
- `forward(params, X)`:
  - Linear combination: $Z = X @ W + b$, shape `(n, 1)`
  - Activation: $A = \sigma(Z)$, shape `(n, 1)`
  - Returns `(A, Cache(X=X, A=A))`
- `mse(A, Y)`:
  - Mean squared error $L = \frac{1}{n} \sum_{k=0}^{n-1} (A_k - Y_k)^2$
  - Must return a Python `float`: `float(np.mean((A - Y) ** 2))`

### Task 2: `backward`
- Four vectorised lines:
  ```python
  n = cache.X.shape[0]
  dA = 2 * (cache.A - Y) / n      # (n, 1)
  dZ = dA * cache.A * (1 - cache.A) # (n, 1)
  dW = cache.X.T @ dZ            # (d_in, 1)
  db = dZ.sum(axis=0)            # (1,)
  ```
- **Shape Trap**: `db` must have shape `(1,)` like `b`. Do NOT pass `keepdims=True` (which returns `(1, 1)`).

### Task 3: `train`
- Implements full-batch gradient descent loop:
  1. Forward pass: `A, cache = forward(params, X)`
  2. Record loss before update: `loss_history.append(mse(A, Y))`
  3. Backward pass: `grads = backward(params, Y, cache)`
  4. In-place parameter updates:
     ```python
     params.W -= lr * grads.dW
     params.b -= lr * grads.db
     ```
  5. Returns `loss_history`.

---

## Experiments

1. **Shape Verification**: Inside your `backward` function, insert:
   ```python
   print(f"X: {cache.X.shape}, A: {cache.A.shape}, dZ: {dZ.shape}, dW: {dW.shape}, db: {db.shape}")
   ```
   Read each shape aloud: e.g. "8 examples by 3 features", "3 features by 1 output".
2. **Batch Scaling Factor $1/n$**:
   Remove the `/ n` from `dA`:
   ```python
   dA = 2 * (cache.A - Y)
   ```
   Observe which tests fail. Then scale the learning rate down by $1/8$ (`lr = lr / 8`). Why does training behave identically when learning rate is scaled by $1/n$?
3. **The Silent Shape Bug**:
   In `train`, flatten $Y$ using `Y = Y.reshape(-1)`.
   Observe that no error is raised, but the loss fails to decrease properly because $A - Y$ expands into an $(n, n)$ outer-product table!
4. **Performance Benchmark**:
   Compare execution speed of Lesson 2's scalar loop implementation vs Lesson 4's matrix implementation on synthetic data ($1\,000$ examples, $20$ features, $200$ epochs). Observe the 50–100× runtime difference.

---

## Self-Check Questions

1. **Loop Elimination**: Where did Lesson 2's loop over examples go? Where did the loop over weights go?
   *(The loop over examples was absorbed into row-wise matrix multiplications and reductions; the loop over weights was absorbed into columns of $W$).*
2. **Mean Division Placement**: Why fold the $\frac{1}{n}$ factor into $dA$ at the beginning of backpropagation rather than dividing $dW$ and $db$ at the end?
   *(Either yields mathematically identical results, but folding $\frac{1}{n}$ into $dA$ ensures that every subsequent intermediate gradient $dZ, dW, db$ is already scaled to the dataset mean).*
3. **Silent Bugs vs Crashes**: Why is a silent broadcasting bug (like `(n, 1) - (n,)`) far more dangerous in machine learning code than a syntax or shape crash?
   *(Because it returns valid floats and runs without error, but trains garbage weights, wasting days of debugging time).*
4. **Parameter Storage**: Why store a single neuron's weights as a 2D matrix of shape `(d_in, 1)` rather than a 1D vector `(d_in,)`?
   *(Because multi-neuron layers in Lesson 5 have shape `(d_in, d_out)`. Keeping a 2D shape allows identical matrix multiplication and backprop code to work unchanged).*
5. **Zero Feature Gradients**: If an entire column of input matrix $X$ is zero, what is the corresponding row of $dW$?
   *(Zero. Since $dW_i = \sum_k X_{ki} \delta_k$, if $X_{ki} = 0$ for all $k$, the weight receives zero gradient).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `db` has shape `(1, 1)` instead of `(1,)` | Used `keepdims=True` in `dZ.sum(axis=0)` | Remove `keepdims=True`: `dZ.sum(axis=0)` yields `(1,)` |
| Gradients off by factor of $n$ | Missing or double-counted factor of $n$ | Fold $\frac{1}{n}$ only once into $dA = 2(A - Y)/n$ |
| Loss stuck around 0.25 on OR3 | Target $Y$ was passed as rank-1 array `(n,)` | Ensure $Y$ has shape `(n, 1)`: `Y.reshape(-1, 1)` |
| `ValueError: matmul` in `backward` | Attempted `X @ dZ` instead of `X.T @ dZ` | Transpose $X$: `dW = cache.X.T @ dZ` |
| Parameters not updating | Subtracted without updating in place | Use `params.W -= lr * grads.dW` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`forward`**: `Z = X @ params.W + params.b`, then `A = sigmoid(Z)`. Return `(A, Cache(X=X, A=A))`.
- **`mse`**: `float(np.mean((A - Y) ** 2))`.
- **`backward`**:
  ```python
  n = cache.X.shape[0]
  dA = 2.0 * (cache.A - Y) / n
  dZ = dA * cache.A * (1.0 - cache.A)
  dW = cache.X.T @ dZ
  db = dZ.sum(axis=0)
  return Grads(dW=dW, db=db)
  ```
- **`train`**: Loop for `_ in range(epochs)`. Compute loss, compute gradients, subtract `lr * grads.dW` from `params.W` and `lr * grads.db` from `params.b`.
</details>
