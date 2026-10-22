# Lesson 1 — Theory: one neuron

Lesson 0 established the calculus tools—derivatives, partial derivatives, and the
chain rule. But calculus needs something to differentiate.

This lesson builds the foundational atomic building block of deep learning:
**the artificial neuron**. We construct it as a typed, reproducible Python class,
explore its forward pass, prove the numerical stability of its sigmoid activation,
and act as the learning algorithm ourselves by hand-tuning weights to compute
fundamental logical gates (OR and AND).

## Learning objectives

By the end of this lesson, you will be able to:
1. Identify and explain the role of each component of an artificial neuron: inputs, weights, bias, weighted sum, and activation.
2. Execute a forward pass calculation by hand, understanding why inactive inputs ($x_i = 0$) silence their corresponding weights.
3. Prove why smooth activations like the sigmoid are necessary for gradient-based learning, unlike non-differentiable step functions.
4. Implement numerically stable sigmoid activations that avoid floating-point overflow for extreme inputs ($z = \pm 1000$).
5. Interpret a neuron's decision boundary geometrically as a hyperplane, using the bias as an adjustable activation threshold.
6. Hand-tune weights and biases to solve linearly separable logical gates (OR, AND) with high confidence.

---

## 1. Concept: a neuron is a mixing desk

### Analogy: the audio mixing console

Imagine an audio engineer balancing a recording session:
- Each instrument has its own dedicated input channel $x_i$.
- Each channel has an adjustable slider knob (the **weight** $w_i$) that controls
  how much of that instrument's signal enters the mix.
- A master slider (the **bias** $b$) raises or lowers the overall volume regardless
  of the individual channel signals.
- A master limiter (the **activation function** $\sigma$) prevents the output signal
  from clipping, squashing the mixed volume into a bounded range between $0$ and $1$.

| Audio mixing desk | Artificial neuron | Mathematical symbol | Code implementation |
|---|---|---|---|
| Instrument channels | Inputs | $x_0, x_1, \dots, x_{d-1}$ | `x: Inputs` (`tuple[float, ...]`) |
| Channel volume sliders | **Weights** | $w_0, w_1, \dots, w_{d-1}$ | `neuron.w: list[float]` |
| Master volume offset | **Bias** | $b$ | `neuron.b: float` |
| Combined acoustic signal | **Weighted sum** | $z = \sum_{i} w_i x_i + b$ | `neuron.weighted_sum(x)` |
| Output limiter / compressor | **Activation** | $a = \sigma(z)$ | `neuron.forward(x)` |
| Final listening decision | **Prediction** | $\hat{y} \in \{0, 1\}$ | `neuron.predict(x)` |

```
   x₀ ──(w₀)──┐
   x₁ ──(w₁)──┼──▶  z = Σ wᵢxᵢ + b  ──▶  a = σ(z)  ──▶  output, between 0 and 1
   x₂ ──(w₂)──┘          ▲
                         b
```

### The core takeaway

The entire "knowledge" or "memory" of an artificial neural network resides exclusively
in its numerical parameters: its **weights and biases**. There are no hidden heuristics,
rules, or decision trees. Learning simply means finding the numerical values of $\mathbf{w}$
and $b$ that produce the desired outputs.

---

## 2. The forward pass

Running input features through a neuron to produce an activation and prediction is
called the **forward pass**:

1. **Calculate the pre-activation weighted sum**:
   $$z = w_0 x_0 + w_1 x_1 + \dots + w_{d-1} x_{d-1} + b = \sum_{i=0}^{d-1} w_i x_i + b$$
2. **Apply the non-linear activation**:
   $$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$
3. **Threshold for binary classification**:
   $$\hat{y} = \begin{cases} 1 & \text{if } a > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Worked example: computing by hand

Consider a neuron with weights $\mathbf{w} = [1.0, -2.0, 0.5]$, bias $b = 0.5$,
and an input vector $\mathbf{x} = (1.0, 0.0, 1.0)$:

| Step | Operation | Formula | Calculation | Result |
|---|---|---|---|---|
| 1 | First input contribution | $w_0 x_0$ | $1.0 \times 1.0$ | $+1.0$ |
| 2 | Second input contribution | $w_1 x_1$ | $-2.0 \times 0.0$ | $0.0$ *(silent input)* |
| 3 | Third input contribution | $w_2 x_2$ | $0.5 \times 1.0$ | $+0.5$ |
| 4 | Add master bias | $+ b$ | $+ 0.5$ | $+0.5$ |
| 5 | **Weighted sum $z$** | $\sum w_i x_i + b$ | $1.0 + 0.0 + 0.5 + 0.5$ | **$2.0$** |
| 6 | **Activation $a$** | $\sigma(2.0)$ | $\frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.1353}$ | **$0.8808$** |
| 7 | **Prediction $\hat{y}$** | $\mathbb{I}(a > 0.5)$ | $0.8808 > 0.5$ | **$1$** |

> [!NOTE]
> Notice row 2: when an input feature $x_i = 0$, its weight is multiplied by zero.
> An inactive feature silences its weight completely.

| Math | Tensor Shape | Code equivalent |
|---|---|---|
| $z = \sum_{i=0}^{d-1} w_i x_i + b$ | scalar float | `neuron.weighted_sum(x)` |
| $a = \sigma(z)$ | scalar float in $(0, 1)$ | `neuron.forward(x)` |
| $\hat{y} = 1 \text{ if } a > 0.5 \text{ else } 0$ | integer $0$ or $1$ | `neuron.predict(x)` |

---

## 3. The sigmoid activation and numerical stability

### The sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
  σ(z)
  1.0 ┼                                 ┌─────────
      │                                ╱
  0.5 ┼───────────────┬───────────────┘
      │              ╱
  0.0 ┼─────────────┴─────────────────────────────
      -6            0                             6  Weighted sum z
```

| $z$ | $-6.0$ | $-2.0$ | $0.0$ | $+2.0$ | $+6.0$ |
|---|---|---|---|---|---|
| $\sigma(z)$ | $0.0025$ | $0.1192$ | $\mathbf{0.5000}$ | $0.8808$ | $0.9975$ |

The activation $a = \sigma(z)$ can be interpreted as the neuron's **degree of confidence**
that the answer is $1$:
- Large positive $z$ ($z > +4$): Activation approaches $1.0$ (confident "yes").
- Large negative $z$ ($z < -4$): Activation approaches $0.0$ (confident "no").
- Center point $z = 0$: Activation is exactly $0.5$ (completely undecided).

### Why not a hard step function?

The earliest artificial neuron, Rosenblatt's 1958 Perceptron, used the Heaviside step:
$H(z) = 1$ if $z > 0$ else $0$.

Why did deep learning replace the step function with the sigmoid?
Because the derivative of a hard step function is **zero everywhere** (except at $z=0$,
where it is undefined). If the slope is zero, gradient descent has no slope to follow:
nudging a weight produces zero change in output, so the neuron cannot learn.

The sigmoid is a **smooth, continuous approximation of a step function**. Because it is
smooth everywhere, it provides non-zero gradients that guide weights toward better solutions.

### The numerical stability trap: preventing `OverflowError`

In Python, 64-bit IEEE 754 floats overflow when an exponent exceeds $\approx 709.78$:
`math.exp(710)` raises `OverflowError: math range error`.

Consider the textbook formula evaluated at $z = -1000$:
$$\sigma(-1000) = \frac{1}{1 + e^{-(-1000)}} = \frac{1}{1 + e^{1000}} \implies \text{CRASH!}$$

Mathematically, $\sigma(-1000)$ should be practically zero. To prevent overflow, we apply
a simple algebraic trick: multiply the numerator and denominator by $e^z$:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^z}{e^z} = \frac{e^z}{e^z + 1}$$

Both mathematical expressions are completely identical. But notice their numerical behavior:
- When $z \ge 0$: $e^{-z} \le 1.0$. Computing $e^{-z}$ can never overflow.
- When $z < 0$: $e^z \le 1.0$. Computing $e^z$ can never overflow. For large negative numbers
  like $z = -1000$, $e^{-1000}$ underflows cleanly to floating-point $0.0$, producing
  $\frac{0.0}{0.0 + 1.0} = \mathbf{0.0}$ without error!

```python
def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        return e / (1.0 + e)
```

Real numerical computing libraries (including SciPy and PyTorch) implement the sigmoid
using this exact dual-branch pattern.

---

## 4. What a single neuron can compute

A neuron predicts $\hat{y} = 1$ when $a > 0.5$, which occurs if and only if $z > 0$:

$$\sum_{i=0}^{d-1} w_i x_i + b > 0 \iff \sum_{i=0}^{d-1} w_i x_i > -b$$

This reveals the geometric role of the bias:
> **The bias $b$ acts as an adjustable threshold or hurdle that the weighted sum must clear.**

### The geometry of the decision boundary

The equation $\mathbf{w} \cdot \mathbf{x} + b = 0$ defines a **hyperplane**:
- In 2D input space, it is a **straight line**: $w_0 x_0 + w_1 x_1 + b = 0$.
- In 3D input space, it is a **flat 2D plane**: $w_0 x_0 + w_1 x_1 + w_2 x_2 + b = 0$.

The neuron partitions input space into two halves: points on one side output $1$,
and points on the other output $0$. Any problem whose classes can be cleanly divided
by a flat cut is called **linearly separable**.

### Hand-tuning weights for boolean gates

For 3-input binary problems ($x_i \in \{0, 1\}$), the 8 possible inputs form the
corners of a 3D unit cube $[0, 1]^3$.

#### 1. The OR3 gate ("at least one input is 1")
Set all three weights equal to $10.0$: $w = [10.0, 10.0, 10.0]$.

| Number of active inputs | Example input $\mathbf{x}$ | Weighted sum $\sum w_i x_i$ | Desired output $y$ |
|---|---|---|---|
| 0 active inputs | $(0, 0, 0)$ | $0.0$ | $0$ (must stay below threshold) |
| 1 active input | $(1, 0, 0)$ | $10.0$ | $1$ (must clear threshold) |
| 2 active inputs | $(1, 1, 0)$ | $20.0$ | $1$ (must clear threshold) |
| 3 active inputs | $(1, 1, 1)$ | $30.0$ | $1$ (must clear threshold) |

We need a threshold between $0.0$ and $10.0$. Setting the threshold at $5.0$ means
$-b = 5.0 \implies b = \mathbf{-5.0}$.
- For $(0, 0, 0)$: $z = 0 - 5 = -5 \implies a = \sigma(-5) = 0.0067 \implies \hat{y} = 0$.
- For any input with at least one $1$: $z \ge 10 - 5 = +5 \implies a \ge \sigma(+5) = 0.9933 \implies \hat{y} = 1$.

#### 2. The AND3 gate ("all three inputs are 1")
With the same weights $w = [10.0, 10.0, 10.0]$:
- Inputs with $0, 1,$ or $2$ active inputs produce sums of $0, 10,$ or $20$ (must be $0$).
- Only $(1, 1, 1)$ produces a sum of $30$ (must be $1$).

We place the threshold between $20.0$ and $30.0$, for instance at $25.0$:
$$-b = 25.0 \implies b = \mathbf{-25.0}$$

#### Why large weights matter: the confidence requirement
Weights of $w = [1.0, 1.0, 1.0]$ with bias $b = -0.5$ also classify OR3 correctly,
but for a single active input $z = 1.0 - 0.5 = 0.5 \implies a = \sigma(0.5) \approx \mathbf{0.62}$.
While $0.62 > 0.5$ produces the right label, the neuron is barely confident.
Scaling weights up to $10.0$ pushes activations to $0.99$ and $0.01$, producing decisive,
confident predictions.

```
        (0,1,1)=1 ─────── (1,1,1)=1
         ╱│                ╱│
   (0,1,0)=1 ─────── (1,1,0)=1
        │ │               │ │
        │(0,0,1)=1 ───────│(1,0,1)=1
        │╱                │╱
   (0,0,0)=0 ─────── (1,0,0)=1
```
*A single flat cut slices off $(0,0,0)$ from the other seven corners on OR3.*

---

## 5. Why a typed class?

In this course, we implement models using modern Python 3.12+ type hints and dataclasses:

```python
@dataclass
class Neuron:
    w: list[float]  # one weight per input, length d_in
    b: float = 0.0
```

### 1. Eliminating dictionary bugs
In untyped code, developers often use dictionaries: `neuron = {"weights": [...], "bias": 0.0}`.
A typo like `neuron["weight"]` fails only at runtime, often deep inside a training loop.
With typed classes, your IDE and type checker (`pyright` in strict mode) catch typos instantly.

### 2. Preventing silent truncation in `zip`
Python's built-in `zip(w, x)` stops as soon as the **shorter** list is exhausted:
```python
w = [1.0, 2.0, 3.0]
x = (4.0, 5.0)
list(zip(w, x))  # returns [(1.0, 4.0), (2.0, 5.0)] — quietly ignoring w[2]!
```
If a 3-weight neuron received a 2-element input, untyped code would quietly ignore
the third weight without any error. In our `weighted_sum` implementation, we explicitly validate:

```python
if len(x) != len(self.w):
    raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
```

### 3. Explicit, reproducible randomness
Functions in this repository never call global random functions like `random.random()`.
Instead, an explicit `np.random.Generator` is passed into every factory function:

```python
Neuron.random(d_in=3, rng=np.random.default_rng(0))
```
Passing the generator guarantees that every experiment, test suite, and debugging run
is $100\%$ reproducible.

---

## Summary reference table

| Concept | Mathematical definition | Shape / Type | Code method |
|---|---|---|---|
| **Weighted sum** | $z = \sum w_i x_i + b$ | float | `neuron.weighted_sum(x)` |
| **Activation** | $a = \sigma(z) = \frac{1}{1 + e^{-z}}$ | float in $(0, 1)$ | `neuron.forward(x)` |
| **Prediction** | $\hat{y} = \mathbb{I}(a > 0.5)$ | integer $0$ or $1$ | `neuron.predict(x)` |
| **Accuracy** | $\frac{1}{n} \sum \mathbb{I}(\hat{y}_k == y_k)$ | float in $[0, 1]$ | `accuracy(neuron, data)` |
| **Safe sigmoid** | $1 / (1 + e^{-z})$ if $z \ge 0$ else $e^z / (1 + e^z)$ | float in $(0, 1)$ | `sigmoid(z)` |
| **OR3 weights** | $w = [10, 10, 10], b = -5$ | `Neuron` | `HAND_OR` |
| **AND3 weights** | $w = [10, 10, 10], b = -25$ | `Neuron` | `HAND_AND` |
