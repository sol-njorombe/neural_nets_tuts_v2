# Lesson 6 — Layers, softmax, minibatches

**Read [THEORY.md](THEORY.md) first.**

## What you build

A modular neural-network library implementing:
- Reusable `Dense`, `ReLU`, and `Sigmoid` layers obeying a uniform layer contract.
- A `Sequential` pipeline container.
- Numerically stable `softmax`, `cross_entropy`, and `cross_entropy_backward`.
- A minibatch training loop (`minibatches`, `train_epoch`, `sgd_step`).

Then you assemble a multi-layer perceptron and train it to read handwritten digits with $> 96\%$ test accuracy.

## Setup

Run the test suite and demonstration script:

```bash
pytest lesson06_layers -x
python lesson06_layers/layers.py
```

To preview the handwritten digits dataset:

```bash
python -c "from nnlib.data import load_digits, show_digit; d = load_digits(); print(f'Label: {d.y_train[0]}'); print(show_digit(d.X_train[0]))"
```

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `Dense.forward`, `Dense.backward` | `forward(X) -> Y`, `backward(dY) -> dX` | [THEORY §2](THEORY.md#2-layers-are-bricks) |
| 2 | `ReLU`, `Sigmoid` | `forward(X) -> Y`, `backward(dY) -> dX` | [THEORY §3](THEORY.md#3-relu-and-vanishing-gradients) |
| 3 | `softmax`, `cross_entropy`, `cross_entropy_backward` | Probability and loss functions | [THEORY §4](THEORY.md#4-multi-class-classification-softmax-and-cross-entropy) |
| 4 | `Sequential.forward`, `Sequential.backward`, `sgd_step` | Container and parameter updates | [THEORY §5](THEORY.md#5-a-network-is-a-list-sequential) |
| 5 | `minibatches`, `train_epoch`, `accuracy` | Training utilities | [THEORY §6](THEORY.md#6-minibatch-stochastic-gradient-descent) |

### Task 1: `Dense`
- `forward(X)`: stores `self.X = X`; returns $X @ W + b$.
- `backward(dY)`:
  - Mutate parameter gradients in place:
    `self.dW[...] = self.X.T @ dY`
    `self.db[...] = dY.sum(axis=0)`
  - Return upstream blame: `return dY @ self.W.T`.

### Task 2: Activation Layers (`ReLU` & `Sigmoid`)
- `ReLU`:
  - `forward(X)`: cache `self.mask = (X > 0)`; return `X * self.mask`.
  - `backward(dY)`: return `dY * self.mask`.
- `Sigmoid`:
  - `forward(X)`: compute and cache `self.A = 1.0 / (1.0 + np.exp(-X))`; return `self.A`.
  - `backward(dY)`: return `dY * self.A * (1.0 - self.A)`.

### Task 3: Softmax and Cross-Entropy
- `softmax(logits)`:
  - Subtract row max for numerical stability: `shifted = logits - logits.max(axis=1, keepdims=True)`.
  - Compute exponentials and divide by row sum with `keepdims=True`.
- `cross_entropy(P, labels)`:
  - Extract true class probabilities with fancy indexing: `P[np.arange(n), labels]`.
  - Clip with `np.clip(p_correct, 1e-15, 1.0)` to avoid $\ln(0)$.
  - Return `float(-np.mean(np.log(p_correct)))`.
- `cross_entropy_backward(P, labels)`:
  - Copy `P`: `dZ = P.copy()`.
  - Subtract 1 from the correct class: `dZ[np.arange(n), labels] -= 1.0`.
  - Divide by batch size $n$: `return dZ / n`.

### Task 4: `Sequential` and `sgd_step`
- `Sequential.forward(X)`: pass through layers in order: `for layer in self.layers: X = layer.forward(X)`.
- `Sequential.backward(dY)`: pass backwards through layers in reverse: `for layer in reversed(self.layers): dY = layer.backward(dY)`.
- `sgd_step(params, lr)`: update each `(param, grad)` array in place:
  ```python
  for p, g in params:
      p -= lr * g
  ```

### Task 5: Training Pipeline
- `minibatches(X, y, batch_size, rng)`:
  - Permute indices: `indices = rng.permutation(len(X))`.
  - Yield slices `(X[batch_idx], y[batch_idx])` for chunks of size `batch_size`.
- `train_epoch(model, X, y, batch_size, lr, rng)`:
  - Loop over minibatches, executing forward $\to$ loss $\to$ backward $\to$ `sgd_step`.
  - Return average training loss across batches.

---

## Experiments

1. **Stacking Depth**:
   Add a second hidden layer to the digit classifier:
   `Sequential([Dense(64, 32, rng), ReLU(), Dense(32, 32, rng), ReLU(), Dense(32, 10, rng)])`.
   Notice how easily depth is configured without modifying any gradient logic.
2. **ReLU vs Sigmoid**:
   Replace `ReLU()` layers with `Sigmoid()` layers in a 4-hidden-layer network. Compare validation accuracy after 5 epochs. Observe how the vanishing gradient problem causes deep sigmoid networks to stall.
3. **Batch Size Dynamics**:
   Compare training with `batch_size=1` (pure SGD), `batch_size=32` (minibatch), and `batch_size=1437` (full batch). Record the seconds per epoch and validation accuracy reached after 5 epochs.
4. **Error Analysis**:
   After training, locate validation examples where `predictions != y_val`. Display them using `show_digit(X_val[i])` to see whether the misclassifications are visually ambiguous digits.

---

## Self-Check Questions

1. **Upstream Blame**: Why does a layer's `backward(dY)` return the gradient with respect to its input $X$, rather than with respect to its parameters $W$ and $b$?
   *(The input gradient $dX = \frac{\partial L}{\partial X}$ is the output blame needed by the preceding layer to continue the chain rule. Parameter gradients are stored internally within the layer).*
2. **State Caching**: Why does `Sigmoid` cache its output $A$, whereas `ReLU` caches a boolean `mask`?
   *(Because $\sigma'(z) = A(1 - A)$ can be computed directly from $A$ without re-evaluating exponentials, whereas $\text{ReLU}'(x)$ depends strictly on whether the input $x$ was positive).*
3. **Loss Geometry**: Why does categorical cross-entropy penalize confident wrong predictions far more aggressively than quadratic loss (MSE)?
   *($-\ln(p) \to \infty$ as $p \to 0$, creating steep gradients that rapidly push the model away from confident errors).*
4. **Computational Efficiency**: With a dataset of 1,437 examples, why is `batch_size=32` much faster in wall-clock time than `batch_size=1`, even though both perform similar amounts of total arithmetic?
   *(Because batch sizes of 32 utilize vectorized BLAS matrix instructions and cache locality, whereas Python loops with `batch_size=1` incur massive interpreter dispatch overhead).*
5. **Composability**: What code would need to change if you wanted to implement a new layer type (such as LayerNorm or Conv2D)?
   *(Only the new layer class implementing `forward(X)` and `backward(dY)`. `Sequential`, optimizers, and the training loop remain completely unchanged).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Training runs without error, but loss does not change | Reassigned arrays instead of in-place mutation (e.g. `p = p - lr * g`) | Use in-place operators: `p -= lr * g` and `self.dW[...] = ...` |
| `nan` or `inf` in softmax | Did not subtract maximum logit before exponentiation | Subtract row max with `keepdims=True`: `logits - logits.max(axis=1, keepdims=True)` |
| Softmax rows do not sum to 1.0 | Missing `keepdims=True` in denominator sum | Use `exp_z.sum(axis=1, keepdims=True)` |
| Deep-stack gradient checks fail, but single layers pass | `Sequential.backward` did not reverse layer order, or failed to pass upstream blame | Iterate using `reversed(self.layers)` and pipe `dY = layer.backward(dY)` |
| Validation accuracy stuck at $\approx 10\%$ (random guess) | Permuted $X$ and $y$ with different random index orders | Use a single permutation array: `indices = rng.permutation(len(X)); X[indices], y[indices]` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`Dense.backward`**:
  ```python
  self.dW[...] = self.X.T @ dY
  self.db[...] = dY.sum(axis=0)
  return dY @ self.W.T
  ```
- **`ReLU`**:
  `forward`: `self.mask = (X > 0.0); return X * self.mask`
  `backward`: `return dY * self.mask`
- **`softmax`**:
  ```python
  shifted = logits - logits.max(axis=1, keepdims=True)
  exp_z = np.exp(shifted)
  return exp_z / exp_z.sum(axis=1, keepdims=True)
  ```
- **`cross_entropy_backward`**:
  ```python
  n = len(labels)
  dZ = P.copy()
  dZ[np.arange(n), labels] -= 1.0
  return dZ / n
  ```
- **`minibatches`**:
  ```python
  n = len(X)
  order = rng.permutation(n)
  for start in range(0, n, batch_size):
      batch_idx = order[start : start + batch_size]
      yield X[batch_idx], y[batch_idx]
  ```
</details>
