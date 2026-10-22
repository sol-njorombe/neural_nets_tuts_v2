# Lesson 8 — Theory: automatic differentiation

Short, and probably the most satisfying lesson. You write about 60 lines and
never derive a gradient by hand again. It is also exactly the idea at the heart
of PyTorch, so lesson 9 will feel familiar.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Explain Reverse-Mode Autodiff**: Describe reverse-mode automatic differentiation as dynamic computational graph construction with local chain-rule closures.
2. **Implement a Computational Node (`Value`)**: Implement arithmetic operator overloading (`+`, `*`, `**`, `exp`, `tanh`, `relu`) that dynamically tracks children and local backward closures.
3. **Prove the Gradient Accumulation Rule**: Explain why `+=` must be used when accumulating gradients into children (multivariable chain rule over branching paths).
4. **Implement Reverse Topological Traversal**: Implement post-order depth-first search to ensure that no node backpropagates blame until all its consumers have contributed to its gradient.
5. **Contrast Scalar vs Tensor Autodiff**: Compare the runtime profile of Python scalar computational graphs with native tensor autograd engines (PyTorch, JAX).

---

## 1. The problem

Every `backward` you have written was the chain rule applied to a handful of
additions, multiplications and activation functions. You had to derive it by
hand for each new kind of layer, and keep it in sync with `forward`. One slip
gives a network that learns badly without crashing; only the gradient checks
caught those.

Lesson 0 §3 listed three ways to get derivatives: numerical (slow, approximate), analytic
(hand-derived, error-prone), and **automatic** (exact, programmatic). This lesson builds the third.

---

## 2. Numbers that keep receipts

Until now, `c = a * b` gave you `c` and discarded its computational provenance. Instead, wrap numbers in a class that keeps a receipt of its operation. A `Value` holds:

| Field | Type | Description |
|---|---|---|
| `data` | `float` | The scalar value itself (forward evaluation) |
| `children` | `tuple[Value, ...]` | Direct ancestor operands in the computational graph |
| `_backward` | `Callable[[], None]` | Local closure passing blame from `self` to its `children` |
| `grad` | `float` | $\frac{\partial \text{loss}}{\partial \text{self}}$, populated during `backward()` |

Executing standard arithmetic **is** the forward pass, dynamically creating a directed acyclic graph (DAG):

```python
x, w, b = Value(2.0), Value(3.0), Value(-4.0)
L = (x * w + b) ** 2      # L.data == 4.0, graph constructed dynamically
L.backward()               # reverse topological graph traversal
print(w.grad)              # 8.0 — computed automatically without hand calculus!
```

```
Computational Graph (Forward → | Backward ◀):
   x (2.0) ─┐
            ├─[*]── u (6.0) ─┐
   w (3.0) ─┘                ├─[+]── s (2.0) ──[**2]── L (4.0)
   b (-4.0) ─────────────────┘
```

---

## 3. Local Chain Rule: Each Operation Knows Only Its Local Slope

No individual operation needs global network context. Each mathematical primitive only needs to know how its output changes with respect to its immediate inputs (derived in Lesson 0):

| Operation | Forward Output | Local Partial Derivative(s) | Backward Gradient Accumulation |
|---|---|---|---|
| `out = a + b` | $a + b$ | $\frac{\partial \text{out}}{\partial a} = 1, \; \frac{\partial \text{out}}{\partial b} = 1$ | `a.grad += 1.0 * out.grad`<br>`b.grad += 1.0 * out.grad` |
| `out = a * b` | $a \cdot b$ | $\frac{\partial \text{out}}{\partial a} = b, \; \frac{\partial \text{out}}{\partial b} = a$ | `a.grad += b.data * out.grad`<br>`b.grad += a.data * out.grad` |
| `out = a ** n` | $a^n$ ($n \in \mathbb{R}$) | $\frac{\partial \text{out}}{\partial a} = n a^{n-1}$ | `a.grad += (n * a.data**(n-1)) * out.grad` |
| `out = exp(a)` | $e^a$ | $\frac{\partial \text{out}}{\partial a} = e^a = \text{out}$ | `a.grad += out.data * out.grad` |
| `out = tanh(a)` | $\tanh(a)$ | $\frac{\partial \text{out}}{\partial a} = 1 - \tanh^2(a)$ | `a.grad += (1.0 - out.data**2) * out.grad` |
| `out = relu(a)` | $\max(0, a)$ | $\frac{\partial \text{out}}{\partial a} = \mathbb{I}[a > 0]$ | `a.grad += (1.0 if a.data > 0 else 0.0) * out.grad` |

In `_backward()`, each child node receives:

$$\Delta \text{child.grad} = \left(\frac{\partial \text{out}}{\partial \text{child}}\right) \cdot \text{out.grad}$$

All other operations derive from these primitives:
- Subtraction: $a - b = a + (-1 \cdot b)$
- Division: $a / b = a \cdot b^{-1}$
- Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$, whose derivative $\sigma(x)(1 - \sigma(x))$ falls out automatically without writing a single line of derivative code!

---

## 4. Running Backpropagation

Calling `L.backward()` on the root scalar loss executes two stages:

1. **Base Seed**: Initializes the root gradient to 1.0 ($\frac{\partial L}{\partial L} = 1.0$). All other nodes begin at `grad = 0.0`.
2. **Reverse Topological Traversal**: Visits every node in reverse topological order, triggering its `_backward()` closure.

### Worked Numerical Example

For $L = (x \cdot w + b)^2$ with $x = 2$, $w = 3$, $b = -4$:

| Stage | Node | Formula | Value |
|---|---|---|---|
| **Forward** | $u$ | $x \cdot w = 2 \times 3$ | $6.0$ |
| | $s$ | $u + b = 6 + (-4)$ | $2.0$ |
| | $L$ | $s^2 = 2^2$ | $4.0$ |
| **Backward** | $L.\text{grad}$ | Seed $\frac{\partial L}{\partial L}$ | **$1.0$** |
| | $s.\text{grad}$ | $\frac{\partial L}{\partial s} = 2s \cdot L.\text{grad} = 2(2)(1)$ | **$4.0$** |
| | $u.\text{grad}$ | $\frac{\partial L}{\partial u} = 1 \cdot s.\text{grad} = 1(4)$ | **$4.0$** |
| | $b.\text{grad}$ | $\frac{\partial L}{\partial b} = 1 \cdot s.\text{grad} = 1(4)$ | **$4.0$** |
| | $w.\text{grad}$ | $\frac{\partial L}{\partial w} = x \cdot u.\text{grad} = 2(4)$ | **$8.0$** |
| | $x.\text{grad}$ | $\frac{\partial L}{\partial x} = w \cdot u.\text{grad} = 3(4)$ | **$12.0$** |

### Two Critical Implementation Requirements

#### 1. Accumulate, Never Overwrite (`+=`)
If a variable is referenced multiple times in an expression (e.g. $y = x + x$), multiple computational paths branch from $x$.
By the multivariable chain rule (Lesson 0 §6):
$$\frac{\partial L}{\partial x} = \sum_{\text{paths } p} \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}\Bigg|_p = 1 + 1 = 2$$
Using `=` instead of `+=` overwrites previous gradients and produces erroneous values.

#### 2. Topological Ordering
A node must not execute its `_backward()` closure until **all** of its downstream consumers have finished propagating their blame into its `.grad`.
Topological sort via post-order depth-first search guarantees this invariant:

```python
order = []
visited = set()

def visit(v):
    if v not in visited:
        visited.add(v)
        for child in v.children:
            visit(child)
        order.append(v)  # appended only after all ancestors are processed

visit(root)
# order[::-1] provides the valid reverse topological execution sequence
```

---

## 5. Arbitrary Architectures with Zero Custom Calculus

`tiny_nn.py` constructs multi-layer perceptrons entirely out of `Value` objects:
```python
# Forward pass:
loss = mse(model(x), y)

# Zero gradients:
for p in model.parameters():
    p.grad = 0.0

# Automatic backward:
loss.backward()

# Parameter step:
for p in model.parameters():
    p.data -= lr * p.grad
```

Notice that adding additional layers, changing loss functions, or switching to $\tanh$ requires zero changes to backward routines.

### Why Must Gradients Be Cleared (`grad = 0.0`)?
Because `_backward()` accumulates gradients via `+=`, failing to reset `p.grad = 0.0` causes gradients from successive training epochs to sum indefinitely. This mirrors PyTorch's mandatory `optimizer.zero_grad()` pattern.

---

## 6. Comparison: Scalar vs Tensor Autodiff

| Dimension | Scalar Autodiff (`Value`) | Tensor Autodiff (PyTorch, JAX) |
|---|---|---|
| **Graph Node** | Single floating-point number | Multidimensional array (`Tensor`) |
| **Node Overhead** | High (Python object allocation per scalar op) | Negligible (one node per batch matrix op) |
| **Execution Engine** | Pure Python recursion | Native C++ / CUDA kernels |
| **Memory Locality** | Pointer graph across Python heap | Contiguous memory buffers on CPU / GPU |
| **Typical Target** | Pedagogical clarity (Karpathy's micrograd) | Production deep learning |
