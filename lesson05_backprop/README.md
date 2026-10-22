# Lesson 5 — Hidden layers and backpropagation

**Read [THEORY.md](THEORY.md) first.**

## What you build

A neural network with a hidden layer that solves the three-input XOR problem starting from random weights — the exact problem Lessons 2 and 4 proved a single neuron can never solve. You first construct a working XOR3 network by hand to prove a solution exists, then train a randomly initialized network using backpropagation.

## Setup

Run the test suite and demonstration script:

```bash
pytest lesson05_backprop -x
python lesson05_backprop/backprop.py
```

## The types

| Type | Fields | Shapes | Role |
|---|---|---|---|
| `TwoLayerParams` | `W1`, `b1`, `W2`, `b2` | `(d_in, d_hidden)`, `(d_hidden,)`, `(d_hidden, 1)`, `(1,)` | Model parameters for both layers |
| `Cache` | `X`, `H`, `A` | `(n, d_in)`, `(n, d_hidden)`, `(n, 1)` | Forward intermediate activations |
| `Grads` | `dW1`, `db1`, `dW2`, `db2` | identical shapes to parameters | Backpropagation gradients |

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `HAND_XOR` | `TwoLayerParams` | [THEORY §2](THEORY.md#2-xor-by-hand) |
| 2 | `forward` | `(params, X) -> (A, Cache)` | [THEORY §3](THEORY.md#3-the-forward-pass) |
| 3 | `backward` | `(params, Y, cache) -> Grads` | [THEORY §4, §5](THEORY.md#4-backpropagation-passing-blame-backwards) |
| 4 | `train` | `(params, X, Y, lr, epochs) -> list[float]` | [THEORY §6](THEORY.md#6-training-dynamics--pitfalls) |

### Task 1: `HAND_XOR`
- Choose constant weights and biases implementing the step-counting logic from [THEORY §2](THEORY.md#2-xor-by-hand):
  - `W1`: shape `(3, 3)` filled with $20.0$.
  - `b1`: shape `(3,)` set to `np.array([-10.0, -30.0, -50.0])`.
  - `W2`: shape `(3, 1)` set to `np.array([[20.0], [-20.0], [20.0]])`.
  - `b2`: shape `(1,)` set to `np.array([-10.0])`.

### Task 2: `forward`
- Hidden layer forward pass:
  - $Z_1 = X @ W_1 + b_1$, shape `(n, d_hidden)`
  - $H = \sigma(Z_1)$, shape `(n, d_hidden)`
- Output layer forward pass:
  - $Z_2 = H @ W_2 + b_2$, shape `(n, 1)`
  - $A = \sigma(Z_2)$, shape `(n, 1)`
- Return `(A, Cache(X=X, H=H, A=A))`.

### Task 3: `backward`
- Compute backpropagation in reverse:
  ```python
  n = cache.X.shape[0]
  # 1. Output layer
  dA  = 2 * (cache.A - Y) / n
  dZ2 = dA * cache.A * (1.0 - cache.A)
  dW2 = cache.H.T @ dZ2
  db2 = dZ2.sum(axis=0)

  # 2. Backpropagate to hidden activations
  dH  = dZ2 @ params.W2.T

  # 3. Hidden layer
  dZ1 = dH * cache.H * (1.0 - cache.H)
  dW1 = cache.X.T @ dZ1
  db1 = dZ1.sum(axis=0)
  ```

### Task 4: `train`
- Iterate over `epochs`:
  - Forward pass and compute MSE loss.
  - Backward pass to obtain `Grads`.
  - Update all four parameter arrays in-place:
    ```python
    params.W1 -= lr * grads.dW1
    params.b1 -= lr * grads.db1
    params.W2 -= lr * grads.dW2
    params.b2 -= lr * grads.db2
    ```

---

## Experiments

1. **Inspect Trained Hidden Features**:
   After training on XOR3, inspect the values of `H` for all 8 input patterns. Does each hidden neuron specialize into an interpretable feature (e.g., detecting specific input counts)?
2. **Architecture Sizing**:
   Train models with $d_{\text{hidden}} \in \{1, 2, 3, 4, 8\}$ across multiple random seeds.
   - Why can $d_{\text{hidden}} = 1$ never achieve > 75% accuracy?
   - Why does $d_{\text{hidden}} = 2$ sometimes get stuck in local minima?
3. **The Symmetry Trap**:
   Initialize all weights to $0.5$ and all biases to $0.0$. Train for 1,000 epochs.
   Inspect the columns of `H`: why are all hidden columns completely identical?
4. **Gradient Attenuation**:
   Compute the mean absolute magnitude `np.abs(grads.dW2).mean()` vs `np.abs(grads.dW1).mean()` during epoch 0. Why is the gradient arriving at Layer 1 smaller than at Layer 2?

---

## Self-Check Questions

1. **Geometric Representation**: How does adding a hidden layer allow the network to solve XOR3 when a single neuron cannot?
   *(Each hidden neuron places a distinct hyperplane cut; the hidden representation maps the 8 vertices of the 3D XOR cube into a space where they are linearly separable).*
2. **Backpropagation Scaling**: In `dH = dZ2 @ W2.T`, why is output error $dZ_2$ multiplied by $W_2^T$?
   *(Because $W_2$ measures how strongly each hidden neuron influenced the output; by the chain rule, a hidden neuron with higher weight bears proportionally greater blame for output error).*
3. **Activation Derivative Confusion**: Why does the hidden layer gradient use $H * (1 - H)$ rather than $A * (1 - A)$?
   *($H$ is the activation of the hidden layer; the derivative of $\sigma(Z_1)$ is $\sigma(Z_1)(1 - \sigma(Z_1)) = H(1 - H)$).*
4. **Symmetry Breaking**: Why must neural network weights be initialized randomly rather than to all zeros or identical constants?
   *(Identical weights produce identical forward activations and identical backward gradients across all hidden units, preventing them from differentiating into distinct feature detectors).*
5. **Architectural Modularity**: If you wanted to add a third layer to this network, which functions would need to be rewritten?
   *(Both `forward` and `backward` would have to be modified manually. This motivates the modular `Layer` abstraction in Lesson 6).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `dW2` is correct, but `dW1` is wrong | Used `A * (1 - A)` for hidden layer derivative | Use hidden activations: `dZ1 = dH * H * (1 - H)` |
| `ValueError` in `dZ2 @ params.W2` | Forgot to transpose `W2` | Transpose `W2`: `dH = dZ2 @ params.W2.T` |
| Network plateaus at 75% accuracy on XOR | Model fell into a local minimum | Use $d_{\text{hidden}} = 4$, $\text{lr} = 2.0$, and verify random seed |
| `HAND_XOR` outputs $\approx 0.5$ | Weights chosen were too small | Scale weights to $20.0$ so sigmoids saturate to 0 or 1 |
| Parameter dimensions mismatched | Biases broadcasted with incorrect shape | Ensure `b1` is `(d_hidden,)` and `b2` is `(1,)` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1 (`HAND_XOR`)**:
  `W1 = np.full((3, 3), 20.0)`
  `b1 = np.array([-10.0, -30.0, -50.0])`
  `W2 = np.array([[20.0], [-20.0], [20.0]])`
  `b2 = np.array([-10.0])`
- **Task 2 (`forward`)**:
  `Z1 = X @ params.W1 + params.b1; H = sigmoid(Z1)`
  `Z2 = H @ params.W2 + params.b2; A = sigmoid(Z2)`
- **Task 3 (`backward`)**:
  Remember to pass blame back using `dH = dZ2 @ params.W2.T`, and apply activation derivative on `cache.H`.
- **Task 4 (`train`)**:
  Subtract `lr * grads` from all four parameter fields in-place.
</details>
