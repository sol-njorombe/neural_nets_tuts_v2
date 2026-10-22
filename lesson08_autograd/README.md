# Lesson 8 — Autograd

**Read [THEORY.md](THEORY.md) first.**

## What you build

A complete reverse-mode automatic differentiation engine from scratch:
- A `Value` class that records computational lineage as standard Python arithmetic expressions are evaluated.
- Local derivative implementations for `__mul__`, `__pow__`, `exp`, `tanh`, and `relu`.
- A directed acyclic graph (DAG) topological sort engine with automatic backpropagation (`backward()`).
- Verification against `tiny_nn.py`, which trains a multi-layer network on XOR without writing a single line of custom calculus.

## Setup

Run the test suite and demonstration script:

```bash
pytest lesson08_autograd -x
python lesson08_autograd/tiny_nn.py
```

---

## Tasks

`__add__` is provided as a complete reference implementation. Study its pattern carefully before proceeding.

| # | What | Target Operation | Theory Reference |
|---|---|---|---|
| 1 | `__mul__` | Operator `*` and `__rmul__` | [THEORY §3](THEORY.md#3-local-chain-rule-each-operation-knows-only-its-local-slope) |
| 2 | `__pow__`, `exp`, `tanh`, `relu` | Power and activation primitives | [THEORY §3](THEORY.md#3-local-chain-rule-each-operation-knows-only-its-local-slope) |
| 3 | `topological_order`, `backward` | Graph traversal and backpropagation | [THEORY §4](THEORY.md#4-running-backpropagation) |

### Task 1: `__mul__`
- Convert second operand to `Value` if it is a plain `int` or `float`: `other = other if isinstance(other, Value) else Value(float(other))`.
- Compute forward result: `out = Value(self.data * other.data, children=(self, other), op="*")`.
- Implement `_backward()` closure:
  ```python
  def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
  out._backward = _backward
  ```

### Task 2: Non-linear and Power Operations
- `__pow__(n)`: supports scalar powers ($n \in \mathbb{R}$). In `_backward`: `self.grad += (n * (self.data ** (n - 1))) * out.grad`.
- `exp()`: forward is `math.exp(self.data)`. In `_backward`: `self.grad += out.data * out.grad`.
- `tanh()`: forward is `math.tanh(self.data)`. In `_backward`: `self.grad += (1.0 - out.data ** 2) * out.grad`.
- `relu()`: forward is `max(0.0, self.data)`. In `_backward`: `self.grad += (1.0 if self.data > 0 else 0.0) * out.grad`.

### Task 3: Graph Traversal and `backward`
- `topological_order(root)`:
  - Recursive depth-first search appending nodes to an ordering list post-order (after visiting children).
  - Maintains a `visited: set[Value]` to ensure each node appears exactly once.
- `Value.backward()`:
  - Seed the root gradient: `self.grad = 1.0`.
  - Traverse `reversed(topological_order(self))`, calling `v._backward()` on each node.

---

## Experiments

1. **Visualize the Computational DAG**:
   Write a small helper function iterating through `topological_order(L)`:
   ```python
   for v in topological_order(L):
       print(f"op={v.op!r:>5} | data={v.data:7.3f} | grad={v.grad:7.3f}")
   ```
   Run this on $L = (x \cdot w + b)^2$ from [THEORY §4](THEORY.md#4-running-backpropagation) and verify that every computed gradient matches hand calculations.
2. **Gradient Accumulation Verification**:
   In `tiny_nn.py`, comment out the line `p.grad = 0.0` before each training step.
   Observe how training diverges because gradients accumulate across successive epochs.
3. **Automatic Sigmoid**:
   Construct sigmoid using arithmetic:
   ```python
   x = Value(0.5)
   s = Value(1.0) / (Value(1.0) + (-x).exp())
   s.backward()
   print(x.grad, s.data * (1 - s.data)) # Both equal 0.2350037
   ```
   Notice that the analytical sigmoid derivative $\sigma(x)(1 - \sigma(x))$ emerges automatically without writing a dedicated backward rule!
4. **Custom Operation**:
   Implement a `.log()` method with derivative $\frac{1}{x}$, and construct binary cross-entropy loss using `Value` objects.
5. **Runtime Profiling**:
   Benchmark `tiny_nn.py` on XOR3 and calculate the approximate wall-clock time required to train on 1,437 MNIST images.

---

## Self-Check Questions

1. **Multivariable Chain Rule**: Where in `Value` is the multivariable chain rule ("paths add") implemented?
   *(In the `+=` operator inside every `_backward` closure. If a node branches into multiple downstream paths, each path contributes additively to its `.grad`).*
2. **Topological Invariant**: Why must the computational graph be sorted into reverse topological order before triggering backpropagation?
   *(Because a node must accumulate all incoming blame from all of its consumers before it passes blame upstream to its children. If visited prematurely, children receive incomplete gradients).*
3. **Seed Condition**: Why does `backward()` set `self.grad = 1.0` as its initial step?
   *(Because $\frac{\partial L}{\partial L} = 1.0$. Without this base case seed, all downstream gradients would remain zero).*
4. **Autonomous Differentiation**: In `tiny_nn.py`, which function derived the gradient of $\tanh$ or the backpropagation equations of the multi-layer network?
   *(Nobody. The chain rule was evaluated entirely by the autograd engine executing local operations in reverse topological order).*
5. **Memory and Speed Bottlenecks**: Why is scalar autograd impractical for large vision models?
   *(Every scalar operation allocates an individual Python object, reference pointers, and a closure on the heap. Modern deep learning requires batched tensor autograd executing on parallel hardware).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| In expression `a + a`, `a.grad` is 1.0 instead of 2.0 | Used `=` instead of `+=` in `_backward()` | Always accumulate gradients: `self.grad += ...` |
| All computed gradients are 0.0 | Forgot to seed root gradient `self.grad = 1.0` in `backward()` | Set `self.grad = 1.0` before topological loop |
| `TypeError: 'Value' object is not iterable` | Wrote `children=(self)` instead of `(self,)` | Use a 1-tuple with trailing comma: `children=(self,)` |
| In expression `b * b`, gradients are incorrect | Node visited multiple times or traversed out of order | Ensure `topological_order` uses a `visited` set and appends post-order |
| `exp()` gradient incorrect | Used `self.data` instead of `out.data` | In `exp`, derivative is $e^x = \text{out.data}$: `self.grad += out.data * out.grad` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`__mul__`**:
  ```python
  other = other if isinstance(other, Value) else Value(float(other))
  out = Value(self.data * other.data, children=(self, other), op="*")
  def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
  out._backward = _backward
  return out
  ```
- **`exp`**:
  ```python
  out = Value(math.exp(self.data), children=(self,), op="exp")
  def _backward():
      self.grad += out.data * out.grad
  out._backward = _backward
  return out
  ```
- **`topological_order`**:
  ```python
  order = []
  visited = set()
  def visit(v: Value) -> None:
      if v not in visited:
          visited.add(v)
          for child in v.children:
              visit(child)
          order.append(v)
  visit(root)
  return order
  ```
</details>
