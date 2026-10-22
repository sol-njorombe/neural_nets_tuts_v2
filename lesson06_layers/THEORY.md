# Lesson 6 — Theory: layers, softmax, minibatches

By the end of this lesson your own code reads handwritten digits at about 97%
accuracy. Three ideas get you there: layers as interchangeable bricks, a loss
for choosing between ten classes, and training on small random batches.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Formulate the Layer Contract**: Define the composable `Layer` protocol (`forward(X)` and `backward(dY)`) and explain why it abstracts away network depth.
2. **Implement Core Network Bricks**: Implement forward and backward passes for `Dense`, `ReLU`, and `Sigmoid` layers following strict shape specifications.
3. **Analyze the Vanishing Gradient Problem**: Explain why sigmoid activations cause gradients to vanish exponentially across deep layers ($0.25^L$) and how `ReLU` prevents this.
4. **Derive Softmax Cross-Entropy Gradients**: Prove that the gradient of categorical cross-entropy with respect to pre-softmax logits simplifies elegantly to $\frac{1}{n}(P - Y_{\text{one-hot}})$.
5. **Implement Minibatch SGD**: Explain minibatch stochastic gradient descent as the optimal balance between full-batch stability and single-example noise.

---

## 1. The data

1,797 images of handwritten digits, each 8×8 pixels. Each image is flattened to
64 numbers between 0 (white) and 1 (dark), and labelled with the digit (0–9) it shows.

```
label: 5          X row: 64 numbers         y: 5
  --@@@@@@%%++
  ++@@##==--::
  **@@
  ++@@--
  ..%%%%..
    ..@@##
    ::@@++
  ++@@**
```

`load_digits()` returns a `Digits` dataclass: 1,437 training images and 360
**validation** images. You never train on the validation images; they are how
you evaluate whether the network learned general digit features or merely memorised its training examples
(Lesson 7 explores generalization in depth).

---

## 2. Layers are bricks

Lesson 5's `backward` was hard-coded for one fixed two-layer architecture. But every layer in a deep network actually performs the exact same operational contract:

```
Forward:
  X (input) ────▶ [ Layer: stores internal state ] ────▶ Y (output)

Backward:
  dX (to prev layer) ◀──── [ Layer: computes parameter grads ] ◀──── dY (from next layer)
```

### The Layer Contract

1. `forward(X)`: takes input activations $X$, stores whatever intermediate state `backward` will require, and returns output activations $Y$.
2. `backward(dY)`: takes output error blame $dY = \frac{\partial L}{\partial Y}$, computes and stores parameter gradients $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$ internally in-place, and returns upstream input blame $dX = \frac{\partial L}{\partial X}$.

A layer's `backward` computes the local chain rule for its own operation and nothing else. It does not need to know what precedes or follows it.

### `Dense` Layer

A fully connected linear transformation $Y = XW + b$:

| Forward | Backward | Shapes |
|---|---|---|
| $Y = X @ W + b$ | $dW = X^T @ dY$ | `(d_in, d_out)`, matching `W` |
| | $db = dY.\text{sum}(\text{axis}=0)$ | `(d_out,)`, matching `b` |
| | $dX = dY @ W^T$ | `(n, d_in)`, matching `X` (returned upstream) |

### Activation Layers as Bricks
Activations are layers with zero parameters. They take $X$, apply a non-linear scalar function $f(X)$, and in `backward(dY)` return $dX = dY \odot f'(X)$. By making activations independent layers, you can swap between `Sigmoid()`, `ReLU()`, or `GELU()` by modifying a single element in a list.

---

## 3. ReLU and Vanishing Gradients

$$\text{ReLU}(x) = \max(0, x), \qquad \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

Why is ReLU the ubiquitous standard for hidden layers in modern deep learning?

Recall from Lesson 0 §5 and Lesson 5 §5 that the sigmoid derivative is:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 0.25$$

When backpropagating through a chain of $L$ sigmoid layers, the gradient passes through repeated multiplications by $\sigma'(z)$:

$$\text{gradient at layer 1} \propto \prod_{l=1}^{L} \sigma'(z_l) \le (0.25)^L$$

- For $L = 5$ layers: $(0.25)^5 \approx 0.00097$ (less than $0.1\%$ of the initial gradient).
- For $L = 10$ layers: $(0.25)^{10} \approx 9.5 \times 10^{-7}$.

Early layers receive negligible signal and stop learning. This is the **vanishing gradient problem**.

In contrast, the derivative of ReLU is exactly $1$ for all positive pre-activations:
$$\text{ReLU}'(x) = 1 \quad \forall x > 0$$
Signals propagate backward through arbitrarily deep networks without geometric decay.

```
Forward:  mask = (X > 0);  A = X * mask
Backward: dX = dY * mask
```

---

## 4. Multi-Class Classification: Softmax and Cross-Entropy

For digit recognition, the final layer produces 10 unconstrained real scores $z \in \mathbb{R}^{10}$, called **logits**.

### Softmax Activation

The **softmax** function maps arbitrary logits $z$ into a valid categorical probability distribution:

$$p_j = \frac{e^{z_j}}{\sum_{k=0}^{C-1} e^{z_k}}, \qquad p_j > 0, \quad \sum_{j=0}^{C-1} p_j = 1$$

#### Numerical Stability Trick
If $z_j = 1000$, $e^{1000} \to \infty$ (`OverflowError` / `inf`).
Because shifting all logits by any constant $c$ leaves probabilities unchanged:

$$\frac{e^{z_j - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c} e^{z_j}}{e^{-c} \sum_k e^{z_k}} = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Setting $c = \max_k z_k$ guarantees that the largest exponent is $e^0 = 1$, preventing numerical overflow.

### Categorical Cross-Entropy Loss

The cross-entropy loss measures the negative log-likelihood assigned to the true label $y \in \{0, \dots, C-1\}$:

$$L = -\frac{1}{n} \sum_{r=0}^{n-1} \ln P[r, y_r]$$

If the model predicts $P[r, y_r] = 0.99$, $-\ln(0.99) \approx 0.010$ (tiny penalty).
If the model assigns $P[r, y_r] = 0.01$, $-\ln(0.01) \approx 4.605$ (massive penalty). Confident mistakes are penalised severely.

### Mathematical Derivation of Softmax + Cross-Entropy Gradient

Consider a single example with logits $z \in \mathbb{R}^C$, probabilities $p = \text{softmax}(z)$, and ground-truth one-hot vector $y^*$.
The loss is $L = -\sum_k y^*_k \ln p_k$.
By the chain rule:

$$\frac{\partial L}{\partial z_i} = \sum_{k} \frac{\partial L}{\partial p_k} \frac{\partial p_k}{\partial z_i} = \sum_k \left(-\frac{y^*_k}{p_k}\right) \frac{\partial p_k}{\partial z_i}$$

Now compute the Jacobian $\frac{\partial p_k}{\partial z_i}$ of softmax:
- If $k = i$:
  $$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \sum_j e^{z_j} - e^{z_i} e^{z_i}}{\left(\sum_j e^{z_j}\right)^2} = p_i - p_i^2 = p_i(1 - p_i)$$
- If $k \ne i$:
  $$\frac{\partial p_k}{\partial z_i} = \frac{0 - e^{z_k} e^{z_i}}{\left(\sum_j e^{z_j}\right)^2} = -p_k p_i$$

Substituting these into the chain rule summation:

$$\frac{\partial L}{\partial z_i} = -\frac{y^*_i}{p_i} p_i(1 - p_i) - \sum_{k \ne i} \frac{y^*_k}{p_k} (-p_k p_i) = -y^*_i(1 - p_i) + \sum_{k \ne i} y^*_k p_i$$
$$= -y^*_i + y^*_i p_i + \sum_{k \ne i} y^*_k p_i = -y^*_i + p_i \sum_{\text{all } k} y^*_k$$

Since $y^*$ is one-hot, $\sum_k y^*_k = 1$. Therefore:

$$\frac{\partial L}{\partial z_i} = p_i - y^*_i$$

Averaging over all $n$ batch examples gives:

$$\frac{\partial L}{\partial \text{logits}} = \frac{1}{n} (P - Y_{\text{one-hot}})$$

This is one of the most elegant results in machine learning: the gradient on the logits is simply the probability error vector divided by $n$.

---

## 5. A network is a list (`Sequential`)

```python
model = Sequential([
    Dense(64, 32, rng),
    ReLU(),
    Dense(32, 10, rng)
])
```

- `Sequential.forward(X)`: runs each layer sequentially forward.
- `Sequential.backward(dY)`: runs each layer in reverse order (`reversed(self.layers)`), piping the returned $dX$ of layer $l$ into layer $l-1$.
- `Sequential.params()`: yields references to `(param_array, grad_array)` for every layer that owns learnable parameters.

### In-Place Array Updates Matter
`params()` returns direct references to the layer's internal NumPy arrays.
- `p -= lr * g` modifies the parameter buffer **in place** (retains array identity).
- `p = p - lr * g` creates a new detached array, leaving the layer's weights unmodified!
Likewise, inside `Dense.backward`, use slice assignment `self.dW[...] = ...` and `self.db[...] = ...` to mutate existing gradient buffers.

---

## 6. Minibatch Stochastic Gradient Descent

| Strategy | Batch Size | Gradients per Epoch | Speed per Step | Gradient Variance |
|---|---|---|---|---|
| **Full-batch GD** | $n = 1,437$ | 1 | Slow | Zero (exact mean) |
| **Pure SGD** | $1$ | $1,437$ | Very slow (no vectorization) | Very high (noisy) |
| **Minibatch SGD** | $32$ | $\approx 45$ | Very fast (BLAS parallel) | Moderate (helpful noise) |

Minibatch SGD provides the best of all worlds:
1. **Hardware Efficiency**: batches of 32–128 saturate SIMD and GPU tensor cores.
2. **Fast Progress**: makes 45 parameter updates per epoch instead of 1.
3. **Escaping Plateaus**: gradient sampling noise helps stochastic descent bounce out of sharp saddles and shallow local minima.
