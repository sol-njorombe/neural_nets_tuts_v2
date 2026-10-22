# Lesson 0 — Theory: derivatives, partial derivatives, the chain rule

Training a neural network boils down to one fundamental question asked millions
of times across billions of parameters:

> *If I turn this knob a microscopic amount, does the error go up or down, and by how much?*

The mathematical answer is a **derivative**. This lesson builds the three calculus
tools needed to answer that question everywhere in deep learning:
1. The **derivative** (sensitivity of a function of one input).
2. The **partial derivative** and the **gradient** (sensitivities when a function has many inputs).
3. The **chain rule** (sensitivities through a sequence of operations, and across multiple paths).

In Section 7, we combine all three into **backpropagation on a one-weight network**,
demonstrating the exact optimization step that every neural network executes during training.

## Learning objectives

By the end of this lesson, you will be able to:
1. Explain a derivative as a local sensitivity ratio without relying on symbolic formulas.
2. Compute numerical derivatives via central differences and explain why quadratic error terms cancel out.
3. Compute partial derivatives by freezing other coordinates and assemble the gradient vector $\nabla f$.
4. Apply the chain rule across composite functions, evaluating the outer derivative at the intermediate value.
5. Apply the multivariable chain rule across branching computational paths ("multiply along, add across").
6. Execute a forward and backward pass on a one-weight network and take a gradient descent step.

---

## 1. A derivative is a sensitivity

### Analogy: the shower tap

Turn a shower tap a tiny amount $\Delta x$, and the water temperature $f(x)$ changes
by $\Delta f$. The derivative is the sensitivity ratio:

$$\text{Sensitivity} = \frac{\Delta \text{temperature}}{\Delta \text{turn}} = \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

- **Large positive**: A touchy tap. A tiny clockwise turn makes the water noticeably hotter.
- **Zero**: A dead zone. Turning the tap slightly makes no difference right now.
- **Negative**: Inverted plumbing. Turning the tap up makes the water colder.

Crucially, sensitivity is a local property **at a specific point**, not a property
of the tap as a whole. The same tap can be touchy near the middle of its dial and
completely unresponsive near the extremes.

### Mathematical definition

For a function $f$ of one real number, the derivative $\frac{df}{dx}$ is the limit
as the nudge shrinks to zero:

$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Geometrically, the derivative is the **slope of the tangent line** to the curve of
$f$ at the point $x$.

---

## 2. Computing it numerically: the central difference

When we do not know an analytical formula for $f$, or when we want an independent
check on human calculus, we can approximate the derivative by nudging $x$ by a tiny
step $h$ and measuring the change.

### One-sided vs. central difference

There are two primary ways to estimate this ratio:

1. **One-sided (forward) difference**:
   $$\frac{df}{dx} \approx \frac{f(x + h) - f(x)}{h}$$
2. **Central difference** (looks symmetrically both ways):
   $$\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}$$

### Why the central difference is dramatically superior: Taylor expansion

By Taylor expansion, any smooth function around $x$ can be written as:

$$f(x + h) = f(x) + f'(x)h + \frac{f''(x)}{2!}h^2 + \frac{f'''(x)}{3!}h^3 + \mathcal{O}(h^4)$$
$$f(x - h) = f(x) - f'(x)h + \frac{f''(x)}{2!}h^2 - \frac{f'''(x)}{3!}h^3 + \mathcal{O}(h^4)$$

**For the one-sided difference**:
$$\frac{f(x + h) - f(x)}{h} = f'(x) + \frac{f''(x)}{2}h + \mathcal{O}(h^2) = f'(x) + \mathcal{O}(h)$$
The leading truncation error is proportional to $h$ (first-order error). If $h = 10^{-5}$,
the error is around $10^{-5}$.

**For the central difference**, subtracting $f(x - h)$ from $f(x + h)$:
$$f(x + h) - f(x - h) = 2 f'(x) h + \frac{f'''(x)}{3} h^3 + \mathcal{O}(h^5)$$

Dividing by $2h$:
$$\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{f'''(x)}{6} h^2 + \mathcal{O}(h^4) = f'(x) + \mathcal{O}(h^2)$$

The quadratic terms $\frac{f''(x)}{2}h^2$ **cancel out entirely**! The leading error
is proportional to $h^2$ (second-order error). If $h = 10^{-5}$, the error drops from
$10^{-5}$ down to $\approx 10^{-10}$.

### Worked example: $f(x) = x^2$ at $x = 1$ with $h = 0.5$

The true analytical derivative is $f'(1) = 2(1) = 2.0$.

| Method | Formula | Calculation | Estimate | Error |
|---|---|---|---|---|
| **One-sided** | $\frac{f(1.5) - f(1.0)}{0.5}$ | $\frac{1.5^2 - 1.0^2}{0.5} = \frac{2.25 - 1.0}{0.5}$ | **2.5** | $+0.5$ (large error) |
| **Central** | $\frac{f(1.5) - f(0.5)}{1.0}$ | $\frac{1.5^2 - 0.5^2}{1.0} = \frac{2.25 - 0.25}{1.0}$ | **2.0** | **0.0** (exact for quadratics!) |

### The floating-point trade-off (catastrophic cancellation)

Why not set $h = 10^{-20}$?
Standard 64-bit floating point numbers (`float` in Python) carry only 53 bits of precision
(about 15–17 decimal digits).

When $h$ is extremely small (e.g. $10^{-15}$), $x + h$ and $x - h$ differ only in their
last few digits. Subtracting them causes **catastrophic cancellation**, discarding almost
all significant digits and leaving floating-point noise.

- $h$ too large ($10^{-1}$): Truncation error dominates.
- $h$ too small ($10^{-15}$): Floating-point subtraction roundoff dominates.
- The sweet spot for 64-bit floats: **$h \approx 10^{-5}$** to **$10^{-6}$**.

```python
def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)
```

The numerical derivative is slow (two function evaluations per input) and approximate,
but it requires no symbolic calculus and works on arbitrary code. This makes it the
perfect **automated referee**: every analytical gradient derived in this course is
tested against it.

---

## 3. Three ways to get a derivative

Throughout machine learning, derivatives are computed in three distinct ways:

| Method | How it works | Strength | Weakness | Where used in course |
|---|---|---|---|---|
| **Numerical** | Nudge inputs by $h$ and measure the ratio | Works on any black-box function | Slow ($2$ evaluations per input), approximate | Lesson 0; test suites (gradient checks) |
| **Analytical** | Derive the exact symbolic formula on paper, code it | Exact, fast ($1$ evaluation for all inputs) | Must be manually re-derived for every model | Lessons 0–7 |
| **Automatic** | Record elementary operations on a graph and apply the chain rule automatically | Exact, fast, requires no hand derivation | Complex engine to build | Lesson 8 (autograd), PyTorch |

In this lesson, you write two exact analytical derivatives by hand:
1. Power rule: $\frac{d}{dx} x^2 = 2x$.
2. The sigmoid derivative: $\frac{d}{dz} \sigma(z) = \sigma(z)(1 - \sigma(z))$ (derived in Section 5).

---

## 4. Partial derivatives: several inputs, nudge one

### Freezing coordinates

A function of one input has a single slope. A function of several inputs has **one
slope per input**.

To compute the slope for input $x_i$, **freeze all other inputs as fixed constants**
and take the single-variable derivative with respect to $x_i$.

**Analogy**: Standing on a mountain ridge, your altitude $f(x, y)$ depends on both
your East-West coordinate $x$ and your North-South coordinate $y$.
- $\frac{\partial f}{\partial x}$ is the slope you feel when stepping East while keeping your North coordinate locked.
- $\frac{\partial f}{\partial y}$ is the slope you feel when stepping North while keeping your East coordinate locked.

### Notation: the curly $\partial$

The curly symbol $\partial$ distinguishes a partial derivative from a total derivative:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_0, \dots, x_i + h, \dots) - f(x_0, \dots, x_i - h, \dots)}{2h}$$

### Worked example: $f(x, y) = x^2 y + 3y$ at $(2, 5)$

At $(2, 5)$, $f(2, 5) = 2^2 \cdot 5 + 3 \cdot 5 = 20 + 15 = 35$.

1. **Compute $\frac{\partial f}{\partial x}$**: Treat $y$ as a constant number.
   - $\frac{\partial}{\partial x}(x^2 y) = 2xy$.
   - $\frac{\partial}{\partial x}(3y) = 0$ (contains no $x$, so it acts as a constant).
   - $\frac{\partial f}{\partial x} = 2xy = 2(2)(5) = \mathbf{20}$.

2. **Compute $\frac{\partial f}{\partial y}$**: Treat $x$ as a constant number.
   - $\frac{\partial}{\partial y}(x^2 y) = x^2(1) = x^2$.
   - $\frac{\partial}{\partial y}(3y) = 3$.
   - $\frac{\partial f}{\partial y} = x^2 + 3 = 2^2 + 3 = \mathbf{7}$.

Check by nudging each input independently by $h = 0.01$:

| Nudge | $f$ after nudge | Actual change $\Delta f$ | Predicted change: $\text{slope} \times h$ |
|---|---|---|---|
| $x: 2 \to 2.01$ ($y$ stays 5) | $2.01^2 \cdot 5 + 3 \cdot 5 = 35.2005$ | $+0.2005$ | $20 \times 0.01 = \mathbf{+0.20}$ |
| $y: 5 \to 5.01$ ($x$ stays 2) | $2^2 \cdot 5.01 + 3 \cdot 5.01 = 35.0700$ | $+0.0700$ | $7 \times 0.01 = \mathbf{+0.07}$ |

### The gradient vector ($\nabla f$)

The **gradient** $\nabla f$ is simply the vector containing all partial derivatives,
arranged in input order:

$$\nabla f(x, y) = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right] = [20, 7]$$

> [!IMPORTANT]
> **The Shape Rule:**
> The gradient vector $\nabla f$ **always has the exact same shape and length as the inputs**.
> In a neural network, the parameters are the inputs to the loss function, so the gradient
> with respect to the weights has the exact same shape as the weights!

| Math | Shape | Code |
|---|---|---|
| $\frac{\partial f}{\partial x_i}$ | scalar float | `partial_derivative(f, args, i)` |
| $\nabla f = [\frac{\partial f}{\partial x_0}, \dots, \frac{\partial f}{\partial x_{n-1}}]$ | length $n$, matching `args` | `numerical_gradient(f, args)` |

---

## 5. The chain rule: several steps, multiply the slopes

### Function composition

Neural networks are built by composing functions: an input $x$ produces an intermediate
activation $u = g(x)$, which then feeds into a downstream operation $y = f(u)$.

How sensitive is the final output $y$ to the initial input $x$?

### Analogy: linked gears

Gear A drives Gear B, and Gear B drives Gear C:
- Gear B rotates $2$ times for every turn of Gear A: $\frac{du}{dx} = 2$.
- Gear C rotates $3$ times for every turn of Gear B: $\frac{dy}{du} = 3$.

How many times does Gear C turn when Gear A turns once?

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 3 \times 2 = \mathbf{6}$$

```
   x ──[ slope = 2 ]──▶ u ──[ slope = 3 ]──▶ y        dy/dx = 2 × 3 = 6
```

### Derivation: local linear scaling

Up close, any smooth function is locally linear:
$$\Delta u \approx g'(x) \cdot \Delta x$$
$$\Delta y \approx f'(u) \cdot \Delta u = f'(u) \cdot \big( g'(x) \cdot \Delta x \big)$$

Dividing both sides by $\Delta x$:

$$\frac{dy}{dx} = f'(u) \cdot g'(x) = \frac{dy}{du} \cdot \frac{du}{dx}$$

> [!CAUTION]
> **The Evaluation Point Trap:**
> The outer derivative $f'(u)$ must be evaluated at **$u = g(x)$** (the intermediate value
> that the forward pass actually produced), **never at $x$**!

### Worked example: $y = (3x + 1)^2$ at $x = 2$

1. **Inner step**: $u = g(x) = 3x + 1$.
   - At $x = 2$: $u = 3(2) + 1 = \mathbf{7}$.
   - Inner derivative: $\frac{du}{dx} = 3$.
2. **Outer step**: $y = f(u) = u^2$.
   - Outer derivative: $\frac{dy}{du} = 2u$.
   - Evaluated at $u = 7$: $\frac{dy}{du} = 2(7) = \mathbf{14}$ (not $2 \times 2 = 4$!).
3. **Chain rule product**:
   $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 14 \times 3 = \mathbf{42}$$

Check by expanding first: $y = 9x^2 + 6x + 1 \implies \frac{dy}{dx} = 18x + 6 = 18(2) + 6 = \mathbf{42}$. ✓

### The sigmoid derivative derived via the chain rule

The sigmoid activation is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

Let $u = 1 + e^{-z}$, so $\sigma = u^{-1}$. We differentiate using the chain rule:

1. **Outer derivative**:
   $$\frac{d}{du}(u^{-1}) = -u^{-2} = -\frac{1}{(1 + e^{-z})^2}$$
2. **Inner derivative**:
   $$\frac{d}{dz}(1 + e^{-z}) = -e^{-z}$$
3. **Multiply**:
   $$\frac{d\sigma}{dz} = \left( -\frac{1}{(1 + e^{-z})^2} \right) \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Now factor the numerator:
$$\frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{1 + e^{-z}} \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right)$$

Because $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$\frac{d\sigma}{dz} = \sigma(z) \cdot (1 - \sigma(z))$$

If we let $a = \sigma(z)$ represent the neuron's activation, then:

$$\frac{da}{dz} = a(1 - a)$$

### Properties of the sigmoid derivative

```
  dσ/dz
  0.25 ┼          ┌───┐          Maximum slope = 0.25 at z = 0
       │         ╱     ╲
  0.10 ┼        ╱       ╲        Saturates to 0 as |z| grows
  0.00 ┼───────┴─────────┴──────
       -6     -2   0   2      6  Input z
```

- **Maximum at $z = 0$**: $\sigma(0) = 0.5 \implies \frac{d\sigma}{dz} = 0.5(0.5) = \mathbf{0.25}$.
- **Vanishing slopes**: When $|z|$ is large (e.g., $z = 6$ or $z = -6$), $a \approx 1$ or $a \approx 0$.
  In both cases, $a(1 - a) \approx 0$. When a neuron is pushed into extreme confidence,
  its gradient drops to zero—a preview of the **vanishing gradient problem**!

| Math | Shape | Code |
|---|---|---|
| $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$ | scalar float | `chain_rule(d_outer, inner, d_inner, x)` |
| $\frac{d\sigma}{dz} = a(1 - a)$ | scalar float | `d_sigmoid(z)` |

---

## 6. Several paths: multiply along, add across

### Multivariable chain rule with branching

When an input variable influences the output along more than one computational path,
each path carries its own chain-rule product. The total sensitivity is the **sum**
of the sensitivities along all paths:

```
              ┌──▶ x = t² ──┐
         t ───┤              ├──▶ z = x · y
              └──▶ y = 3t ──┘
```

$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}$$

$$\text{Total derivative} = \sum_{\text{paths}} \prod_{\text{links in path}} \text{local slopes}$$

### Worked example: at $t = 2$

At $t = 2$: $x = 2^2 = 4$ and $y = 3(2) = 6$. The output is $z = 4 \times 6 = 24$.

1. **Path 1 (via $x$)**:
   - $\frac{\partial z}{\partial x} = y = 6$
   - $\frac{dx}{dt} = 2t = 4$
   - Product along Path 1: $6 \times 4 = \mathbf{24}$

2. **Path 2 (via $y$)**:
   - $\frac{\partial z}{\partial y} = x = 4$
   - $\frac{dy}{dt} = 3$
   - Product along Path 2: $4 \times 3 = \mathbf{12}$

3. **Total derivative**:
   $$\frac{dz}{dt} = 24 + 12 = \mathbf{36}$$

Check by direct substitution: $z = (t^2)(3t) = 3t^3 \implies \frac{dz}{dt} = 9t^2 = 9(2^2) = \mathbf{36}$. ✓

### Why this is the core of backpropagation

The golden rule of backpropagation is:
> **Multiply slopes along each path; add contributions across paths.**

- In Lesson 2, a weight influences the loss on every training example: we average (add) across examples.
- In Lesson 5, a hidden neuron connects to multiple output neurons: the errors from all downstream neurons add up.
- In Lesson 8, automatic differentiation accumulates gradients across arbitrary branching graphs using this exact rule.

| Math | Shape | Code |
|---|---|---|
| $\sum_{\text{paths}} \prod \text{slopes}$ | scalar float | `sum_over_paths([[6, 4], [4, 3]])` |

---

## 7. Putting it together: backpropagation on one weight

We now combine sensitivity, analytical derivatives, the chain rule, and branching
into a complete forward and backward pass on the smallest possible neural network:
one input $x$, one weight $w$, a sigmoid activation $\sigma$, and a squared error loss $L$:

```
            Forward pass ──────────────────────────────────────────▶

   w ──▶ [ z = w · x ] ──▶ [ a = σ(z) ] ──▶ [ L = (a − y)² ]
   x ───────┘                                    y ───┘

            ◀────────────────────────────────────────── Backward pass
               dz/dw = x       da/dz = a(1 − a)     dL/da = 2(a − y)
```

### The forward pass: compute and cache
Given $w = 0.5, x = 2.0, y = 1.0$:
1. $z = w \cdot x = 0.5 \times 2.0 = \mathbf{1.0}$
2. $a = \sigma(z) = \sigma(1.0) = \frac{1}{1 + e^{-1}} \approx \mathbf{0.7311}$
3. $L = (a - y)^2 = (0.7311 - 1.0)^2 = (-0.2689)^2 \approx \mathbf{0.0723}$

We store $z, a, L$ in a `Trace` dataclass.

### The backward pass: multiply the local slopes
To find $\frac{\partial L}{\partial w}$, we step backward from the loss to the weight:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

| Backward link | Local derivative formula | Evaluation using cached forward trace | Value |
|---|---|---|---|
| Link 1 ($\frac{\partial L}{\partial a}$) | $2(a - y)$ | $2(0.7311 - 1.0)$ | **$-0.5379$** |
| Link 2 ($\frac{\partial a}{\partial z}$) | $a(1 - a)$ | $0.7311 \times (1 - 0.7311)$ | **$+0.1966$** |
| Link 3 ($\frac{\partial z}{\partial w}$) | $x$ | $2.0$ | **$+2.0000$** |
| **Total gradient $\frac{\partial L}{\partial w}$** | **Product of all 3** | $(-0.5379) \times (0.1966) \times (2.0)$ | **$-0.2115$** |

### Taking a gradient descent step

The gradient is negative ($\frac{\partial L}{\partial w} = -0.2115$), meaning increasing $w$
will decrease the loss. To step downhill, we subtract a fraction $\eta = 0.5$ of the gradient:

$$w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w} = 0.5 - 0.5 \times (-0.2115) = \mathbf{0.6058}$$

Evaluate the new loss at $w_{\text{new}} = 0.6058$:
- $z_{\text{new}} = 0.6058 \times 2.0 = 1.2116$
- $a_{\text{new}} = \sigma(1.2116) \approx 0.7706$
- $L_{\text{new}} = (0.7706 - 1.0)^2 \approx \mathbf{0.0526}$ (down from $0.0723$!)

Repeating this single calculation is **gradient descent**.

### Why save the forward pass values?

Notice that `tiny_backward` needs $a$ and $x$. Because the forward pass cached $a$ in
`Trace`, the backward pass never recomputes the exponential $\sigma(z)$.
This split—**forward pass caches activations; backward pass consumes them**—is the
fundamental architecture of every modern deep learning framework from PyTorch to JAX.

| Math | Shape | Code |
|---|---|---|
| $\frac{\partial L}{\partial w} = 2(a - y) \cdot a(1 - a) \cdot x$ | scalar float | `tiny_backward(tiny_forward(w, x, y), x, y)` |

---

## Summary reference table

| Concept | Mathematical definition | Key rule / formula | Code equivalent |
|---|---|---|---|
| **Numerical derivative** | $\lim_{h \to 0} \frac{f(x+h) - f(x-h)}{2h}$ | Central difference cancels $h^2$ error: $\mathcal{O}(h^2)$ | `numerical_derivative(f, x, h)` |
| **Sigmoid derivative** | $\frac{d}{dz} \sigma(z)$ | $a(1 - a)$, where $a = \sigma(z)$ | `d_sigmoid(z)` |
| **Chain rule** | $\frac{d}{dx} f(g(x))$ | $f'(g(x)) \cdot g'(x)$ (outer evaluated at inner output) | `chain_rule(d_outer, inner, d_inner, x)` |
| **Partial derivative** | $\frac{\partial f}{\partial x_i}$ | Freeze all $x_{j \neq i}$ as constants | `partial_derivative(f, args, i, h)` |
| **Gradient vector** | $\nabla f$ | Vector of all partials; same shape as inputs | `numerical_gradient(f, args, h)` |
| **Multivariable chain** | $\frac{dz}{dt}$ along branches | $\sum_{\text{paths}} \prod \text{slopes}$ | `sum_over_paths(paths)` |
| **Backward pass** | $\frac{\partial L}{\partial w}$ | $\frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = 2(a - y)a(1-a)x$ | `tiny_backward(trace, x, y)` |
| **Gradient step** | $w_{\text{new}}$ | $w - \eta \frac{\partial L}{\partial w}$ | `w - lr * grad` |

