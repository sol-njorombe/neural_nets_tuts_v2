# Lesson 2 — Theory: loss and gradient descent

Lesson 1 ended with you acting as the learning algorithm, manually choosing
weights for simple gates like OR and AND. That approach quickly fails for
complex problems: a network with thousands or millions of parameters cannot be
tuned by human guesswork.

This lesson builds the self-tuning loop that powers modern neural networks:
1. **The loss**: a single smooth number measuring how wrong the neuron's predictions are.
2. **Gradient descent**: an efficient calculus-based method to adjust every knob in the direction of lower loss.
3. **The linear separability limit**: why a single neuron can solve OR and AND, but hits a mathematical wall on XOR.

---

## 1. The loss: measuring wrongness

### Why accuracy cannot guide learning: the plateau problem

In Lesson 1, we measured performance using **accuracy** (the fraction of correct
predictions). While accuracy is the ultimate metric humans care about, it is
unusable as an optimization guide during training.

A neuron's discrete prediction $\hat{y} \in \{0, 1\}$ is formed by thresholding its
smooth activation $a$:

$$\hat{y} = \begin{cases} 1 & \text{if } a > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Suppose a weight is slightly nudged, moving $a$ from $0.51$ to $0.54$. In both
cases, $\hat{y} = 1$. The accuracy does not change at all. The derivative of
accuracy with respect to any weight is **zero almost everywhere**:

$$\frac{\partial \text{Accuracy}}{\partial w_i} = 0$$

If the slope is zero, the optimizer feels a completely flat plateau—there is no
downhill gradient to follow.

```
Accuracy
  1.0 ┼               ┌───────────────
      │               │
  0.5 ┼───────┬───────┘   Flat plateaus: slope = 0 everywhere
      │       │           Nudging weights gives zero signal
  0.0 ┼───────┴───────────────────────
      0      0.5     1.0  Activation a
```

### Mean Squared Error (MSE): a smooth compass

To guide optimization, we need a **loss function** $L$ that changes smoothly with
every tiny nudge of the weights. We compute the gap between the neuron's
activation $a$ and the ground-truth target $y$, square it, and average over the
entire dataset of $n$ examples:

$$L = \frac{1}{n} \sum_{k=1}^n (a_k - y_k)^2$$

Where $k$ indexes the $n$ examples in the training set.

MSE has four essential mathematical properties:
1. **Smoothness and differentiability**: The parabola $(a - y)^2$ is continuous
   with a linear, well-behaved derivative $\frac{\partial L}{\partial a} = 2(a - y)$.
2. **Non-negativity**: $(a - y)^2 \ge 0$. The minimum possible loss is $L = 0$,
   achieved only when the neuron outputs the exact target for every example.
3. **Quadratic penalty**: Large mistakes are punished disproportionately more than
   small uncertainties. An error of $0.8$ contributes $(0.8)^2 = 0.64$, whereas an
   error of $0.1$ contributes only $(0.1)^2 = 0.01$.
4. **The clueless baseline ($L = 0.25$)**: A neuron with all weights and bias set to
   zero outputs $a = 0.5$ for every input. If the true targets are binary $y \in \{0, 1\}$,
   the squared error for every row is:
   $$(0.5 - 1)^2 = 0.25 \quad \text{and} \quad (0.5 - 0)^2 = 0.25$$
   Regardless of whether the problem is OR, AND, or XOR, a clueless neuron produces
   a loss of exactly $0.25$. When inspecting training logs, $L \approx 0.25$ indicates
   that the neuron is performing no better than random guessing.

| Math | Shape | Code |
|---|---|---|
| $(a - y)^2$ | one number | `squared_error(a, y)` |
| $L = \frac{1}{n} \sum_{k=1}^n (a_k - y_k)^2$ | one number | `loss(neuron, data)` |

---

## 2. Trial and error: wiggling and the scaling wall

How do we minimize $L$? The most basic strategy requires no calculus: **coordinate search**
(implemented in the given `train_by_wiggling`).

### The wiggling algorithm
For each parameter (each weight $w_i$, then the bias $b$):
1. Nudge the knob up by a fixed step $h$: $\theta \leftarrow \theta + h$.
2. Measure the loss across the entire dataset. If it improved, keep the new value.
3. If not, try nudging down: $\theta \leftarrow \theta - h$, and measure the loss again.
4. If neither direction helped, restore the original knob value.
5. Move to the next parameter and repeat for several rounds.

### The computational wall

Wiggling works on tiny problems, but it hits an immediate scaling wall.

Suppose a neuron has $d_{\text{in}}$ inputs and $1$ bias, giving $K = d_{\text{in}} + 1$
tunable knobs. Measuring the loss requires passing all $n$ data examples through the neuron:
- Testing $+h$ requires $1$ full pass over the dataset.
- Testing $-h$ (when $+h$ fails) requires a second full pass over the dataset.
- In the worst or average case, evaluating one round of adjustments requires up to
  **$2(d_{\text{in}} + 1)$ passes over the data**.

| Problem | Knobs ($K$) | Passes per round (wiggling) | Passes per round (gradient descent) |
|---|---|---|---|
| OR3 ($d_{\text{in}} = 3$) | 4 | up to 8 passes | 2 passes (1 loss + 1 gradient) |
| 10-input boolean gate | 11 | up to 22 passes | 2 passes |
| Small image layer ($1{,}000$ inputs) | 1,001 | up to 2,002 passes | 2 passes |
| Small deep network ($10^6$ parameters) | 1,000,000 | up to 2,000,000 passes | 2 passes |

Furthermore, wiggling only gives a binary decision ("did the loss go down?"). It has no
knowledge of **how steep** the slope is: it takes the same blind step size whether it is
on a gentle slope or a sheer cliff.

---

## 3. Gradient descent: walking downhill in fog

### Analogy: the hillside in thick fog

Imagine standing on an unfamiliar hillside in dense fog. You want to reach the
valley floor (the minimum loss), but you cannot see more than an inch ahead.
You cannot look across the terrain to spot the valley, but you can feel the slope
of the ground directly under your feet.

The strategy:
1. Feel which direction tilts steepest downhill.
2. Take a step in that direction.
3. Repeat until the ground beneath your feet is flat.

| Hillside | Neural network training | Symbol |
|---|---|---|
| Your coordinates on the hill | Current weights and bias | $w_0, w_1, \dots, b$ |
| Altitude | Value of the loss function $L$ | $L$ |
| Slope under your feet | The **gradient vector** $\nabla L$ | $[\partial L/\partial w_0, \dots, \partial L/\partial b]$ |
| Stride length | The **learning rate** | $\eta$ (`lr`) |
| Taking a downhill step | Parameter update | $\theta \leftarrow \theta - \eta \cdot \partial L/\partial \theta$ |

### Why the minus sign?

The gradient $\nabla L$ mathematically points in the direction of **steepest ascent**
(the direction that increases the loss the fastest).

Because our objective is to **minimize** the loss, we must travel in the exact
opposite direction: $-\nabla L$.

$$\text{knob} \leftarrow \text{knob} - \eta \cdot \frac{\partial L}{\partial \text{knob}}$$

- If the slope $\frac{\partial L}{\partial w_i} > 0$: increasing the weight increases the
  loss. The minus sign reduces $w_i$.
- If the slope $\frac{\partial L}{\partial w_i} < 0$: increasing the weight decreases the
  loss. The minus sign increases $w_i$.

**The breakthrough**: Using calculus and the chain rule, a **single pass** over the
dataset computes the exact slope for every parameter simultaneously.

---

## 4. The analytical gradient of one neuron

We now apply the chain rule from Lesson 0 to derive the exact gradient for a neuron.

### The forward computation graph

For a single data example $(\mathbf{x}, y)$, the forward computation is a chain of operations:

```
        wᵢ ──┐
             ├──▶ z = Σ wᵢxᵢ + b ──▶ a = σ(z) ──▶ Lₖ = (a − y)²
   xᵢ, b ────┘
```

1. **Weighted sum**: $z = \sum_{i=1}^{d_{\text{in}}} w_i x_i + b$
2. **Activation**: $a = \sigma(z) = \frac{1}{1 + e^{-z}}$
3. **Squared error**: $L_k = (a - y)^2$

### The 3-link backward chain

To find how the loss changes when we adjust a weight $w_i$, we differentiate back
through each link in the chain:

$$\frac{\partial L_k}{\partial w_i} = \frac{\partial L_k}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_i}$$

Let us evaluate each link individually:

1. **Link 1 (Loss sensitivity to activation)**:
   $$\frac{\partial L_k}{\partial a} = \frac{d}{da} (a - y)^2 = 2(a - y)$$

2. **Link 2 (Activation sensitivity to weighted sum)**:
   From Lesson 0 §5, the derivative of the sigmoid $\sigma(z)$ with respect to $z$ is:
   $$\frac{\partial a}{\partial z} = \sigma(z)(1 - \sigma(z)) = a(1 - a)$$

3. **Link 3 (Weighted sum sensitivity to parameters)**:
   For any weight $w_i$:
   $$\frac{\partial z}{\partial w_i} = x_i$$
   For the bias $b$:
   $$\frac{\partial z}{\partial b} = 1$$

### The error signal ($\delta$)

Notice that Link 1 and Link 2 are identical for every parameter in the neuron.
We multiply them together and define the result as **$\delta$** ("delta"):

$$\delta = \frac{\partial L_k}{\partial z} = \frac{\partial L_k}{\partial a} \cdot \frac{\partial a}{\partial z} = 2(a - y) \cdot a(1 - a)$$

$\delta$ measures the sensitivity of the loss to the pre-activation sum $z$.
Notice that $\delta \to 0$ in two distinct situations:
- **Confident and correct**: When the prediction matches the target ($a \approx y$),
  the error term $(a - y) \approx 0$, so $\delta \approx 0$.
- **Saturation (flat regions of the sigmoid)**: When $|z|$ is large, $a$ is close to $0$
  or $1$. In both limits, $a(1 - a) \approx 0$, so $\delta \approx 0$. Even if the neuron
  is wrong, if it is extremely confident in its mistake, the gradient shrinks to zero.

### Credit assignment and the input gating rule

Combining $\delta$ with Link 3 yields the parameter gradients for example $k$:

$$\frac{\partial L_k}{\partial w_i} = \delta \cdot x_i$$
$$\frac{\partial L_k}{\partial b} = \delta \cdot 1 = \delta$$

> [!IMPORTANT]
> **The Input Gating Rule:**
> Each weight's gradient is directly proportional to its input feature $x_i$:
> - If $x_i = 0$, then $\frac{\partial L_k}{\partial w_i} = 0$.
> - If an input was silent ($0$), its weight contributed nothing to $z$. It bears
>   **zero responsibility** for the error and receives **zero gradient update** on that example!

### Averaging across the dataset

The total loss $L$ is the arithmetic mean across all $n$ examples: $L = \frac{1}{n} \sum_{k=1}^n L_k$.
Because differentiation is linear, the dataset gradient is the average of the per-example gradients:

$$\frac{\partial L}{\partial w_i} = \frac{1}{n} \sum_{k=1}^n \delta_k x_{ki}$$
$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{k=1}^n \delta_k$$

| Math | Shape | Code |
|---|---|---|
| $\delta_k = 2(a_k - y_k) \cdot a_k(1 - a_k)$ | one float per example | inside loop in `gradients` |
| $\frac{\partial L}{\partial w_i} = \frac{1}{n} \sum_{k=1}^n \delta_k x_{ki}$ | length $d_{\text{in}}$, like `Neuron.w` | `grads.dw` |
| $\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{k=1}^n \delta_k$ | one float | `grads.db` |

### Worked example: a zero neuron on OR3

Consider a 3-input neuron with all weights and bias set to zero ($w = [0, 0, 0], b = 0$)
evaluating the OR3 dataset ($n = 8$ examples).

For every example:
- $z = 0 \cdot x_0 + 0 \cdot x_1 + 0 \cdot x_2 + 0 = 0$
- $a = \sigma(0) = 0.5$
- $a(1 - a) = 0.5 \times 0.5 = 0.25$

Now compute $\delta_k = 2(0.5 - y_k) \cdot 0.25 = 0.5(0.5 - y_k)$:

| Row | Inputs $(x_0, x_1, x_2)$ | Target $y$ | $\delta = 0.5(0.5 - y)$ |
|---|---|---|---|
| 0 | $(0, 0, 0)$ | 0 | $0.5(0.5 - 0) = \mathbf{+0.25}$ |
| 1 | $(0, 0, 1)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 2 | $(0, 1, 0)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 3 | $(0, 1, 1)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 4 | $(1, 0, 0)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 5 | $(1, 0, 1)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 6 | $(1, 1, 0)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |
| 7 | $(1, 1, 1)$ | 1 | $0.5(0.5 - 1) = \mathbf{-0.25}$ |

**Computing $\frac{\partial L}{\partial w_0}$:**
$x_0 = 1$ only on rows 4, 5, 6, 7 (all of which have $y = 1$ and $\delta = -0.25$).
On rows 0–3, $x_0 = 0$, so their contribution is zero.

$$\sum_{k=0}^7 \delta_k x_{k0} = 4 \times (-0.25) = -1.0$$
$$\frac{\partial L}{\partial w_0} = \frac{-1.0}{8} = \mathbf{-0.125}$$

By symmetry, $\frac{\partial L}{\partial w_1} = -0.125$ and $\frac{\partial L}{\partial w_2} = -0.125$.

**Computing $\frac{\partial L}{\partial b}$:**
The bias connects to every row ($x_{\text{bias}} = 1$):

$$\sum_{k=0}^7 \delta_k = 1 \times (+0.25) + 7 \times (-0.25) = 0.25 - 1.75 = -1.50$$
$$\frac{\partial L}{\partial b} = \frac{-1.50}{8} = \mathbf{-0.1875}$$

**The update step (with $\eta = 1.0$):**
$$w_i \leftarrow 0 - 1.0 \times (-0.125) = \mathbf{0.125}$$
$$b \leftarrow 0 - 1.0 \times (-0.1875) = \mathbf{0.1875}$$

Both the weights and bias increase. This makes complete intuitive sense: OR3 outputs $1$
on $7$ out of $8$ rows, so the neuron immediately shifts its weights and threshold toward
producing larger activations.

---

## 5. The training loop and learning rates

### The epoch cycle

One full pass over the dataset followed by a gradient step is an **epoch**.
Training consists of repeating this cycle:

```
repeat for each epoch:
    record L                       # measure loss BEFORE the step (to track progress)
    compute ∇L                     # one pass over data to accumulate exact dw and db
    update parameters              # wᵢ ← wᵢ − η · dwᵢ,   b ← b − η · db
```

Recording $L$ *before* the step guarantees that `history[0]` represents the baseline
loss of the untrained neuron before any adjustments occur.

### Learning rate dynamics

The learning rate $\eta$ (`lr`) is a hyperparameter that scales the step size:

```
Loss L
 │         \               /
 │          \   Too large /
 │           \ (Overshoot)
 │            \  ◄───►   /
 │  Too small  \   *    /   Ideal step
 │  (Crawls)    \______/    (Lands near bottom)
 └────────────────────────────► Parameter θ
```

| Learning rate $\eta$ | Observed behavior |
|---|---|
| **Too small** (e.g. $0.001$) | Loss falls monotonically, but by fractions of a percent per epoch. Training crawls. |
| **Well-tuned** (e.g. $1.0$ – $2.0$) | Loss drops rapidly over the first few hundred epochs, then cleanly levels off near zero. |
| **Too large** (e.g. $50.0$+) | Steps overshoot the valley floor, bouncing wildly between opposite walls or diverging. |

### Gradient checking: the independent referee

How do you guarantee that your analytical gradient formulas in `gradients()` do not
contain subtle sign errors or missing factors of $n$?

We use the central difference formula from Lesson 0 as an independent numerical check:

$$\left(\frac{\partial L}{\partial \theta}\right)_{\text{numerical}} \approx \frac{L(\theta + h) - L(\theta - h)}{2h}$$

Implemented in `numerical_gradients()`, this acts as an automated referee. The test
suite runs this check across multiple random seeds and datasets, verifying that:

$$\left| \left(\frac{\partial L}{\partial \theta}\right)_{\text{analytical}} - \left(\frac{\partial L}{\partial \theta}\right)_{\text{numerical}} \right| < 10^{-7}$$

---

## 6. The wall: why one neuron cannot learn XOR

Train your neuron on `OR3`, `AND3`, or `FIRST_INPUT`, and it will rapidly reach $100\%$
accuracy with loss near zero.

Now train it on `XOR3`. The loss refuses to budge, hovering stubbornly around $0.25$
(the clueless guessing baseline), and accuracy never exceeds $50\%$.

### Geometric proof: linear separability

In Lesson 1 §4, we showed that a neuron's decision boundary is defined by the equation:

$$w_0 x_0 + w_1 x_1 + w_2 x_2 + b = 0$$

In a 3D input space, this equation represents a **flat 2D plane**. The neuron predicts $1$
on one side of the plane and $0$ on the other. A dataset is **linearly separable** if and
only if a single flat plane can divide all $1$ targets from all $0$ targets.

- For **OR3**, only $(0, 0, 0)$ is $0$; the other seven corners are $1$. A single slice
  cuts off the $(0, 0, 0)$ corner cleanly.
- For **AND3**, only $(1, 1, 1)$ is $1$. A single slice cuts off the $(1, 1, 1)$ corner.
- For **XOR3**, the output is $1$ if an odd number of inputs are $1$, and $0$ otherwise.

```
            (0,1,1)=0 ─────── (1,1,1)=1
             ╱│                ╱│
       (0,1,0)=1 ─────── (1,1,0)=0
            │ │               │ │
            │(0,0,1)=1 ───────│(1,0,1)=0
            │╱                │╱
       (0,0,0)=0 ─────── (1,0,0)=1
```

Look at the alternating vertices along each edge of the cube:
- The four $0$ corners are: $(0,0,0), (0,1,1), (1,0,1), (1,1,0)$
- The four $1$ corners are: $(0,0,1), (0,1,0), (1,0,0), (1,1,1)$

Every single edge connects a $0$ to a $1$. Any flat plane that cuts the cube will
inevitably group some $0$s with some $1$s. **No single flat plane can separate these
two sets.**

This is not a failure of gradient descent. Gradient descent is functioning flawlessly:
given the impossible task of separating alternating corners with a single plane, the
optimal mathematical compromise is to place the plane through the center, outputting
$a \approx 0.5$ everywhere, which yields a loss of $(0.5 - y)^2 = 0.25$.

### Historical significance: the first AI winter

In 1969, Marvin Minsky and Seymour Papert published the book *Perceptrons*. They proved
mathematically that single-layer perceptrons could not compute non-linear functions like
exclusive-OR (XOR). Because researchers at the time did not know how to train networks
with multiple layers, funding dried up and the field entered the first "AI Winter" for
more than a decade.

### The road ahead: hidden layers and matrix shapes

To solve XOR, we need **layers of neurons**:
1. A **hidden layer** of neurons each makes a different linear cut across the input space.
2. An **output neuron** combines those cuts, effectively carving out a non-linear decision region.

This solution was popularized in 1986 by Rumelhart, Hinton, and Williams with
**backpropagation**, which you will build in Lesson 5 to solve XOR once and for all.

Before building hidden layers, we face a major engineering hurdle: writing multiple
layers of neurons in nested Python `for` loops is unbearably slow. In **Lesson 3**,
we master NumPy array shapes and matrix multiplication. In **Lesson 4**, we vectorize
a single neuron. Then in **Lesson 5**, we combine them to train deep networks.

---

## Summary reference table

| Concept | Mathematical formula | Shape | Code implementation |
|---|---|---|---|
| **Squared error** | $(a - y)^2$ | scalar float | `squared_error(a, y)` |
| **MSE loss** | $L = \frac{1}{n} \sum (a_k - y_k)^2$ | scalar float | `loss(neuron, data)` |
| **Error signal** | $\delta = 2(a - y) \cdot a(1 - a)$ | scalar float | `delta = 2 * (a - e.y) * a * (1 - a)` |
| **Weight gradient** | $\frac{\partial L}{\partial w_i} = \frac{1}{n} \sum \delta_k x_{ki}$ | length $d_{\text{in}}$ | `grads.dw` |
| **Bias gradient** | $\frac{\partial L}{\partial b} = \frac{1}{n} \sum \delta_k$ | scalar float | `grads.db` |
| **Gradient step** | $\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$ | in-place | `step(neuron, grads, lr)` |
| **Epoch loop** | record $L \to$ gradient $\to$ step | list of floats | `train(neuron, data, lr, epochs)` |
| **Gradient check** | $\frac{L(\theta + h) - L(\theta - h)}{2h}$ | matches $\nabla L$ | `numerical_gradients(neuron, data)` |
