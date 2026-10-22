# Lesson 5 — Theory: hidden layers and backpropagation

This is the lesson the first half of the course has been building towards: the
three-input XOR network, starting from random weights, trained with
backpropagation. With lessons 3 and 4 behind you it is two more lines going
forward and four more going back.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Explain Representation Transformation**: Explain how a hidden layer re-represents input data to resolve linearly non-separable problems like XOR3.
2. **Implement Two-Layer Forward Passes**: Compute batched forward propagation through hidden and output layers, tracking dimensions ($Z_1, H, Z_2, A$).
3. **Derive Multilayer Backpropagation**: Derive the backpropagation algorithm for a 2-layer network, specifically proving how output error $dZ_2$ propagates back to hidden activations through $dH = dZ_2 @ W_2^T$.
4. **Generalise the Backprop Template**: State the universal 4-line backward gradient algorithm executed by each neural network layer.
5. **Diagnose Optimization Phenomena**: Analyze why neural networks suffer from symmetry traps (when initialized with identical weights) and local minima.

---

## 1. A hidden layer is a team of specialists

One neuron makes one flat hyperplane cut across input space. Give the problem to a small team of hidden neurons instead:

```
 inputs (d_in=3)       hidden layer (d_hidden=3)     output (d_out=1)
                       ┌─▶ h₀ ──┐
  x₀ ──┐               │        │
  x₁ ──┼── fully ──────┼─▶ h₁ ──┼──── linear ──────▶ a
  x₂ ──┘   connected   │        │     combination
                       └─▶ h₂ ──┘
```

Each **hidden** neuron looks at the raw inputs and makes its own decision boundary. The
**output** neuron never sees the raw inputs; it sees only the hidden neurons'
activations and combines their opinions. Several flat cuts combined can carve out convex polytopes and complex non-linear decision regions that no single hyperplane can partition.

They are termed "hidden" because the training labels provide targets only for the final output $A$, never for intermediate states $H$. Training must autonomously discover useful feature representations for each hidden specialist.

---

## 2. XOR by hand

Before training with gradient descent, construct a working XOR3 network by hand.
Three hidden specialists:

| Hidden neuron | Fires ($\approx 1$) when | Weights $(w_0, w_1, w_2)$ | Bias $b$ |
|---|---|---|---|
| $h_0$ | at least one input is on ($\ge 1$) | `[20.0, 20.0, 20.0]` | $-10.0$ |
| $h_1$ | at least two inputs are on ($\ge 2$) | `[20.0, 20.0, 20.0]` | $-30.0$ |
| $h_2$ | all three inputs are on ($= 3$) | `[20.0, 20.0, 20.0]` | $-50.0$ |

With weights of 20, the weighted sum $\sum_i x_i w_i$ takes values $0, 20, 40,$ or $60$. Placing biases at $-10, -30, -50$ sets the decision thresholds midway between successive integer counts. Because $|z| \ge 10$, $\sigma(z)$ outputs either $\sigma(-10) \approx 0.000045$ or $\sigma(+10) \approx 0.999955$.

Now, combine the hidden neuron outputs through alternating signs $h_0 - h_1 + h_2$:

| Inputs on ($\sum x_i$) | $h_0$ | $h_1$ | $h_2$ | Linear combination $h_0 - h_1 + h_2$ | Target $\text{XOR3}$ |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | **0** |
| 1 | 1 | 0 | 0 | 1 | **1** |
| 2 | 1 | 1 | 0 | 0 | **0** |
| 3 | 1 | 1 | 1 | 1 | **1** |

To produce the final prediction, the output neuron applies weights $W_2 = [20, -20, 20]^T$ and bias $b_2 = -10$:
- When $h_0 - h_1 + h_2 = 1$, $z_2 = 20(1) - 10 = +10 \implies a = \sigma(10) \approx 1$.
- When $h_0 - h_1 + h_2 = 0$, $z_2 = 20(0) - 10 = -10 \implies a = \sigma(-10) \approx 0$.

The hidden layer has mapped the non-linearly separable 3D parity problem into a clean, linearly separable internal representation $H$.

---

## 3. The forward pass

Both layers are batched linear transformations followed by elementwise activations:

```
Forward graph:
X (n, d_in) ──▶ [ @ W1 + b1 ] ──▶ Z1 (n, d_hidden) ──▶ [ σ ] ──▶ H (n, d_hidden)
H (n, d_hidden) ──▶ [ @ W2 + b2 ] ──▶ Z2 (n, 1)    ──▶ [ σ ] ──▶ A (n, 1)
```

In matrix notation:
$$Z_1 = X W_1 + b_1 \in \mathbb{R}^{n \times d_{\text{hidden}}}, \qquad H = \sigma(Z_1) \in \mathbb{R}^{n \times d_{\text{hidden}}}$$
$$Z_2 = H W_2 + b_2 \in \mathbb{R}^{n \times 1}, \qquad A = \sigma(Z_2) \in \mathbb{R}^{n \times 1}$$

| Math | Array Shape | Code |
|---|---|---|
| $Z_1 = X W_1 + b_1$ | `(n, d_in) @ (d_in, d_hidden) + (d_hidden,) → (n, d_hidden)` | `Z1 = X @ W1 + b1` |
| $H = \sigma(Z_1)$ | `(n, d_hidden)` | `H = sigmoid(Z1)` |
| $Z_2 = H W_2 + b_2$ | `(n, d_hidden) @ (d_hidden, 1) + (1,) → (n, 1)` | `Z2 = H @ W2 + b2` |
| $A = \sigma(Z_2)$ | `(n, 1)` | `A = sigmoid(Z2)` |

---

## 4. Backpropagation: passing blame backwards

### Step 1: Output Layer Gradients
The output layer treats $H$ as its input features. This is identical to Lesson 4:

```python
dA  = 2 * (A - Y) / n             # (n, 1)
dZ2 = dA * A * (1 - A)            # (n, 1)   δ₂ for the output neuron
dW2 = H.T @ dZ2                   # (d_hidden, 1)
db2 = dZ2.sum(axis=0)             # (1,)
```

### Step 2: Passing Blame to the Hidden Layer ($dH$)
By the chain rule, how does a change in hidden activation $H_{rj}$ influence the scalar loss $L$?
Hidden activation $H_{rj}$ directly affects output pre-activation $Z2_{r0} = \sum_k H_{rk} W2_{k0} + b2_0$ with slope $W2_{j0}$.

$$\frac{\partial L}{\partial H_{rj}} = \frac{\partial L}{\partial Z2_{r0}} \frac{\partial Z2_{r0}}{\partial H_{rj}} = dZ2_{r0} \cdot W2_{j0}$$

In matrix form across all batch rows and hidden neurons:
$$dH = dZ_2 @ W_2^T \in \mathbb{R}^{n \times d_{\text{hidden}}}$$

Shapes match: `(n, 1) @ (1, d_hidden) → (n, d_hidden)`.
Each hidden neuron receives blame proportional to its incoming contribution $W_2$ to the output error.

### Step 3: Hidden Layer Gradients
With $dH$ in hand, backpropagate through the hidden sigmoid activation:

$$\delta_{1, rj} = dZ1_{rj} = \frac{\partial L}{\partial H_{rj}} \cdot \sigma'(Z1_{rj}) = dH_{rj} \cdot H_{rj}(1 - H_{rj})$$

Then compute parameter gradients for Layer 1 using input $X$:

$$dW_1 = X^T @ dZ_1 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{hidden}}}$$
$$db_1 = dZ_1.\text{sum}(\text{axis}=0) \in \mathbb{R}^{d_{\text{hidden}}}$$

### Universal 4-Line Layer Backpropagation Template

Every fully connected layer in any deep network follows this universal four-line template in reverse order:

```
δ              = dA * activation_derivative(Z)   # local pre-activation error
dW             = input.T @ δ                    # parameter weight gradient
db             = δ.sum(axis=0)                  # parameter bias gradient
blame_on_input = δ @ W.T                        # passed upstream to previous layer
```

| Math | Array Shape | Code |
|---|---|---|
| $\frac{\partial L}{\partial H} = \delta_2 W_2^T$ | `(n, 1) @ (1, d_hidden) → (n, d_hidden)` | `dH = dZ2 @ W2.T` |
| $\delta_1 = \frac{\partial L}{\partial H} \odot H(1 - H)$ | `(n, d_hidden)` | `dZ1 = dH * H * (1 - H)` |
| $\frac{\partial L}{\partial W_1} = X^T \delta_1$ | `(d_in, d_hidden)` | `dW1 = X.T @ dZ1` |
| $\frac{\partial L}{\partial b_1} = \sum_r \delta_{1, r}$ | `(d_hidden,)` | `db1 = dZ1.sum(axis=0)` |

---

## 5. Worked numerical example

A minimal network: 1 input ($x=1$), 1 hidden neuron ($W_1=[[1]]$, $b_1=0$), 1 output neuron ($W_2=[[2]]$, $b_2=0$), target $y=1$:

| Pass | Step | Expression | Numerical Value |
|---|---|---|---|
| **Forward** | $z_1$ | $1 \cdot 1 + 0$ | $1.0000$ |
| | $h$ | $\sigma(1.0)$ | $0.7311$ |
| | $z_2$ | $0.7311 \cdot 2 + 0$ | $1.4621$ |
| | $a$ | $\sigma(1.4621)$ | $0.8119$ |
| | $L$ | $(0.8119 - 1.0)^2$ | $0.0354$ |
| **Backward** | $dA$ | $2(0.8119 - 1) / 1$ | $-0.3763$ |
| | $dZ_2$ | $-0.3763 \times 0.8119 \times (1 - 0.8119)$ | $-0.0575$ |
| | **$dW_2$** | $h \cdot dZ_2 = 0.7311 \times (-0.0575)$ | **$-0.0420$** |
| | $dH$ | $dZ_2 \cdot W_2 = -0.0575 \times 2.0$ | $-0.1150$ |
| | $dZ_1$ | $dH \cdot h(1 - h) = -0.1150 \times 0.7311 \times 0.2689$ | $-0.0226$ |
| | **$dW_1$** | $x \cdot dZ_1 = 1.0 \times (-0.0226)$ | **$-0.0226$** |

Notice that $|dW_1| < |dW_2|$: because $\sigma'(z) \le 0.25$, backpropagating through multiple sigmoid activations exponentially diminishes the gradient magnitude. This is the **vanishing gradient problem**, which motivates the ReLU activation in Lesson 6.

---

## 6. Training Dynamics & Pitfalls

Training iterates the forward-backward cycle across all four parameter arrays ($W_1, b_1, W_2, b_2$).
Two critical non-bug phenomena occur:

### 1. Symmetry Breaking
If $W_1$ is initialized with identical constant values (e.g. all 0.5):
- Every hidden neuron computes the exact same activation $H_{rj}$ for all $j$.
- In the backward pass, every hidden neuron receives the exact same gradient.
- Every hidden neuron updates identically, collapsing a layer of $d_{\text{hidden}}$ neurons into a single redundant neuron!
**Fix**: Break symmetry with random initialization (`rng.uniform(-1, 1)` or Xavier init).

### 2. Local Minima
With small architectures (e.g., $d_{\text{hidden}} = 2$), gradient descent frequently gets trapped in sub-optimal local minima (e.g., 75% accuracy on XOR3). Expanding the hidden layer to $d_{\text{hidden}} = 4$ creates higher-dimensional escape routes, allowing gradient descent to converge to 100% accuracy reliably.
