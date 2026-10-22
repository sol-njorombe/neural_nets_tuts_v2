# Lesson 0 — Derivatives, partial derivatives, the chain rule

**Read [THEORY.md](THEORY.md) first.** Each task below points directly at the theory section it implements.

## What you build

Six small, focused functions in `derivatives.py` that compute derivatives three ways
(numerical, analytical, and via the chain rule), concluding with backpropagation on a
one-weight network. Everything is written in pure Python using `float` arithmetic—no
NumPy or external libraries yet.

## Setup

Run the test suite as you work through each task:

```bash
pytest lesson00_derivatives -x              # stop on first failure
pytest lesson00_derivatives -k chain        # run only chain-rule tests
```

Every test initially fails with `NotImplementedError: TODO`. Work through the tasks in
sequential order; the test suite is structured to match this sequence.

---

## Tasks

| # | Function(s) | Theory | What it does |
|---|---|---|---|
| 1 | `numerical_derivative` | §2 | Central difference estimate: $(f(x + h) - f(x - h)) / 2h$. The referee for the course. |
| 2 | `d_square`, `d_sigmoid` | §3, §5 | Exact analytical derivatives: $2x$ and $a(1 - a)$, verified against Task 1. |
| 3 | `chain_rule` | §5 | Function composition: $f'(g(x)) \cdot g'(x)$. Evaluates outer slope at $g(x)$. |
| 4 | `partial_derivative`, `numerical_gradient` | §4 | Freeze coordinates to compute $\frac{\partial f}{\partial x_i}$; collect into gradient vector $\nabla f$. |
| 5 | `sum_over_paths` | §6 | Branching calculus: multiply along each path, then sum across paths. |
| 6 | `tiny_forward`, `tiny_backward` | §7 | Backprop on one weight: forward caches values in `Trace`; backward multiplies local slopes. |

### Task 1: The central difference (`numerical_derivative`)
- Implements $\frac{f(x + h) - f(x - h)}{2h}$ with default $h = 10^{-5}$.
- Evaluates $f$ symmetrically on both sides of $x$ so that quadratic error terms cancel out ($\mathcal{O}(h^2)$ precision).

### Task 2: Analytical derivatives (`d_square` and `d_sigmoid`)
- `d_square(x)` returns $2x$.
- `d_sigmoid(z)` computes the exact slope of the sigmoid activation:
  $$a = \sigma(z) = \frac{1}{1 + e^{-z}} \implies \frac{d\sigma}{dz} = a(1 - a)$$
  Note: for negative $z$, use the numerically safe form $e^z / (1 + e^z)$ provided by `sigmoid(z)`.

### Task 3: The chain rule (`chain_rule`)
- Takes `d_outer`, `inner`, `d_inner`, and `x`.
- Evaluates `inner(x)` first to obtain $u = g(x)$.
- Evaluates `d_outer(u)` and multiplies by `d_inner(x)`.

### Task 4: Multivariable gradients (`partial_derivative` and `numerical_gradient`)
- `partial_derivative(f, args, i, h)`: Creates copies of `args`, nudges only index $i$ by $+h$ and $-h$, and computes the central difference.
  **Do not mutate the caller's arguments.**
- `numerical_gradient(f, args, h)`: Computes `partial_derivative` for every index $i \in \{0, \dots, \text{len}(\text{args}) - 1\}$ and returns the gradient list $\nabla f$.

### Task 5: Branching paths (`sum_over_paths`)
- Takes a list of paths, where each path is a sequence of local derivatives along a route:
  $$\text{Total slope} = \sum_{\text{paths}} \prod_{\text{slopes in path}} s$$
- An empty list of paths returns `0.0`.

### Task 6: Backprop on one weight (`tiny_forward` and `tiny_backward`)
- `tiny_forward(w, x, y)`:
  1. $z = w \cdot x$
  2. $a = \sigma(z)$
  3. $L = (a - y)^2$
  Returns a `Trace(z=z, a=a, L=L)`.
- `tiny_backward(trace, x, y)`:
  Uses the cached `trace.a` to compute the three local links:
  - $\frac{\partial L}{\partial a} = 2(a - y)$
  - $\frac{\partial a}{\partial z} = a(1 - a)$
  - $\frac{\partial z}{\partial w} = x$
  Returns their product: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$.

---

## Experiments

Run an interactive Python session to develop intuition:

1. **Floating-point limits of nudging (Catastrophic Cancellation):**
   The true derivative of $x^3$ at $x = 2$ is $3(2^2) = 12.0$.
   ```python
   from derivatives import numerical_derivative
   for h in (1e-1, 1e-3, 1e-5, 1e-8, 1e-11, 1e-14):
       print(f"h={h:1.0e} -> {numerical_derivative(lambda x: x**3, 2.0, h):.10f}")
   ```
   Notice that as $h$ drops below $10^{-11}$, the approximation deteriorates due to subtractive cancellation of floating-point numbers.

2. **Gradient descent by hand:**
   Start at $w = 0.5, x = 2.0, y = 1.0$ and repeatedly step against the gradient:
   ```python
   from derivatives import tiny_forward, tiny_backward
   w, x, y, lr = 0.5, 2.0, 1.0, 0.5
   for step in range(10):
       trace = tiny_forward(w, x, y)
       grad = tiny_backward(trace, x, y)
       w -= lr * grad
       print(f"step {step}: w={w:.4f}, loss={trace.L:.6f}, grad={grad:.4f}")
   ```
   Observe how the step size naturally shrinks as the loss approaches zero.

3. **The vanishing gradient:**
   Try the same loop starting from $w = -5.0$:
   ```python
   w = -5.0
   trace = tiny_forward(w, x, y)
   grad = tiny_backward(trace, x, y)
   print(f"w={w}, loss={trace.L:.4f}, da/dz={trace.a * (1 - trace.a):.6f}, grad={grad:.6f}")
   ```
   Even though the loss is large, the gradient is nearly zero because the sigmoid is saturated ($a \approx 0 \implies a(1 - a) \approx 0$).

---

## The limit

You can now compute how any knob affects any downstream loss. However, this toy setup
features only a single scalar weight, a single input, no bias, and no dataset.

In **Lesson 1**, we structure a real artificial neuron with multiple inputs, adjustable
weights, an additive bias threshold, and evaluation across truth tables.

---

## Self-check & discussion questions

Test your mental model:
1. **Zero derivative**: If $\frac{df}{dx} = 0$ at a point, does that mean $f$ is flat everywhere? In a neural network, what does a zero gradient for a weight indicate?
2. **Numerical referee**: Why do we trust numerical differentiation as an independent test referee even though it is approximate?
3. **Multiplication vs. Addition**: In multivariable branching (§6), what would go wrong mathematically if we multiplied the two path contributions instead of adding them?
4. **Saturation**: In Experiment 3, when $w = -5.0$, the loss is high but the gradient is tiny. Is this a code bug or a mathematical property of sigmoids?
5. **Trace caching**: `tiny_backward` accepts a cached `Trace` instead of re-evaluating the forward pass. What would be the computational penalty of recomputing forward activations in a 50-layer deep network?
6. **Flashcard check**:
   - If at $x = 3$ the slope is $-2$, by roughly how much does $f$ change if $x$ increases by $0.01$? *(Falls by $\approx 0.02$.)*
   - For $f(x, y) = 5x + y^2$, what is $\nabla f$ at $(1, 3)$? *($[5, 6]$.)*
   - If $\frac{\partial L}{\partial w} > 0$, should $w$ be increased or decreased to lower the loss? *(Decreased: step against the gradient.)*

---

## Common mistakes & troubleshooting

| Test failure / symptom | Likely cause | Fix |
|---|---|---|
| `chain_rule` gives 18 instead of 42 | Evaluated `d_outer(x)` instead of `d_outer(inner(x))` | Evaluate outer derivative at the intermediate output $u = g(x)$ |
| `partial_derivative` passes alone, but "leaves inputs unchanged" fails | Mutated `args` in place | Make explicit copies: `up = list(args)` and `down = list(args)` |
| `d_sigmoid` off by a sign | Wrote `a * (a - 1)` | Use $a(1 - a)$ |
| `tiny_backward` sign inverted | Wrote `2 * (y - a)` | The loss derivative $\frac{\partial L}{\partial a} = 2(a - y)$ |
| `numerical_derivative` error is large ($\approx 2.5$) | Implemented one-sided difference $\frac{f(x+h)-f(x)}{h}$ | Use central difference: $\frac{f(x+h)-f(x-h)}{2h}$ |

---

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1:** `(f(x + h) - f(x - h)) / (2 * h)`
- **Task 2:** For `d_sigmoid(z)`, compute `a = sigmoid(z)` and return `a * (1.0 - a)`.
- **Task 3:** Call `d_outer(inner(x))` multiplied by `d_inner(x)`. Do not evaluate `d_outer` at `x`.
- **Task 4:** Use `up = list(args)` and `down = list(args)`. Nudge `up[i] += h` and `down[i] -= h`.
- **Task 5:** Use `math.prod(path)` for each path and sum the products. An empty list should return `0.0`.
- **Task 6:**
  - In `tiny_backward`, read `trace.a`. Do not call `sigmoid()` again.
  - Compute `dL_da = 2 * (trace.a - y)`, `da_dz = trace.a * (1 - trace.a)`, `dz_dw = x`.
  - Return `dL_da * da_dz * dz_dw`.
</details>


