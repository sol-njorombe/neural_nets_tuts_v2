# Lesson 1 — One neuron

**Read [THEORY.md](THEORY.md) first.** Each task below points directly at the theory section it implements.

## What you build

A typed `Neuron` class that processes multidimensional inputs, computes a weighted
sum, applies a numerically stable sigmoid activation, and produces thresholded
predictions. Then you act as the learning algorithm yourself, choosing weights and
biases by hand to compute the logical OR and AND functions.

## Setup

Run the test suite as you work through each task:

```bash
pytest lesson01_neuron -x              # stop on first failure
python lesson01_neuron/neuron.py        # after task 4: run the random neuron demo
```

---

## The data

Each dataset is a `list[Example]`. An `Example` is an immutable dataclass containing:
- `x: Inputs`: the input features as a tuple of floats, e.g. `(1.0, 0.0, 1.0)`.
- `y: int`: the binary ground-truth target, `0` or `1`.

Four truth-table datasets are provided in `nnlib.scalar`, each listing all 8 combinations
of 3-bit binary inputs:

| Name | Rule for target $y = 1$ | Linearly separable? |
|---|---|---|
| `FIRST_INPUT` | The first input $x_0$ is 1 (the other two are distractions) | Yes |
| `OR3` | At least one input is 1 ($\sum x_i \ge 1$) | Yes |
| `AND3` | All three inputs are 1 ($\sum x_i == 3$) | Yes |
| `XOR3` | An odd number of inputs are 1 ($\sum x_i \pmod 2 == 1$) | **No** (The wall) |

---

## Tasks

| # | Function / Class | Theory | What it does |
|---|---|---|---|
| 1 | `sigmoid(z)` | §3 | Numerically safe sigmoid: handles $z = \pm 1000$ without overflow. |
| 2 | `Neuron.random(d_in, rng)` | §5 | Factory method: initializes random weights from `rng.uniform(-1, 1)` and $b = 0$. |
| 3 | `weighted_sum`, `forward`, `predict` | §2 | Forward pass: $z = \sum w_i x_i + b$, $a = \sigma(z)$, and $\hat{y} = \mathbb{I}(a > 0.5)$. |
| 4 | `accuracy(neuron, data)` | — | Evaluates fraction of correct predictions across a dataset: $\frac{1}{n} \sum \mathbb{I}(\hat{y} == y)$. |
| 5 | `HAND_OR`, `HAND_AND` | §4 | Choose weights and bias by hand to solve OR3 and AND3 with high confidence. |

### Task 1: Numerically safe sigmoid (`sigmoid`)
- Compute $\sigma(z) = \frac{1}{1 + e^{-z}}$ for $z \ge 0$.
- Compute $\sigma(z) = \frac{e^z}{1 + e^z}$ for $z < 0$.
- Passing $z = -1000.0$ must return $0.0$ without raising `OverflowError`.

### Task 2: Random initialization (`Neuron.random`)
- Draw $d_{\text{in}}$ weights using `float(rng.uniform(-1, 1))`.
- Set initial bias $b = 0.0$.
- Passing the same `rng` state must produce identical, reproducible weights.

### Task 3: The forward pass methods
- `weighted_sum(x)`:
  - First verify `len(x) == len(self.w)`. If not, raise `ValueError`.
  - Accumulate $z = \text{self.b} + \sum w_i x_i$.
- `forward(x)`:
  - Returns `sigmoid(self.weighted_sum(x))`.
- `predict(x)`:
  - Returns `1` if `self.forward(x) > 0.5`, else `0`.

### Task 4: Scoring accuracy (`accuracy`)
- Count rows where `neuron.predict(example.x) == example.y`.
- Divide by the total number of examples `len(data)`.

### Task 5: Hand-tuning weights (`HAND_OR` and `HAND_AND`)
- Choose weights $w_0, w_1, w_2$ and bias $b$ that classify all 8 rows correctly.
- Test `test_your_neurons_are_confident` requires decisive predictions: $|a - y| < 0.1$
  for every row. Use sufficiently large weights (e.g. $10.0$) so activations commit near $1.0$ and $0.0$.

---

## Running the demo

Once Tasks 1–4 are complete, run:

```bash
python lesson01_neuron/neuron.py
```

Expected output:

```text
Neuron(w=[0.27387593197926163, 0.4304593450974011, 0.20552675214328784], b=0.0)
(0.0, 0.0, 0.0) want 0 got 0.5
(0.0, 0.0, 1.0) want 1 got 0.551
(0.0, 1.0, 0.0) want 1 got 0.606
(0.0, 1.0, 1.0) want 0 got 0.654
(1.0, 0.0, 0.0) want 1 got 0.568
(1.0, 0.0, 1.0) want 0 got 0.618
(1.0, 1.0, 0.0) want 0 got 0.669
(1.0, 1.0, 1.0) want 1 got 0.713
accuracy on XOR3: 0.5
```

Notice that a random neuron is clueless and gets only $50\%$ on XOR3.

---

## Experiments

1. **Random initialization scan:**
   In `neuron.py`, instantiate 10 random neurons with different seeds (`range(10)`).
   Does any random configuration accidentally solve XOR3?
2. **Confidence threshold:**
   For `HAND_OR`, keep all three weights equal and gradually scale them down: $10.0 \to 5.0 \to 2.0 \to 1.0 \to 0.5$.
   At what weight scale does `test_your_neurons_are_confident` fail?
3. **Designing custom logic gates:**
   - Create a neuron that computes a **majority voter**: outputs 1 if at least two inputs are active.
     (What threshold is needed with weights of $10.0$?)
   - Create a neuron that computes: "first input is ON and second input is OFF".
4. **The XOR challenge:**
   Spend up to 5 minutes attempting to find weights and a bias by hand that solve `XOR3`.
   Review THEORY §4 to explain why no single flat plane can separate alternating corners.

---

## The limit

A single neuron can compute OR, AND, and linear gates, but you had to choose the weights
by hand. A modern vision or language network contains millions of weights, and nobody
can configure them manually.

To make the neuron learn its own weights, we need:
1. A smooth metric for error (**loss**).
2. An automated method to adjust weights downhill (**gradient descent**).

That is the subject of **Lesson 2**.

---

## Self-check & discussion questions

Test your mental model:
1. **Silent inputs**: If an input feature is $0.0$ across every example in the dataset, what can its corresponding weight learn? *(Nothing: it is always multiplied by zero.)*
2. **The bias superpower**: What capability does the bias grant a neuron that weights alone cannot provide? *(It allows the neuron to output a non-zero threshold even when all input features are zero.)*
3. **Decisive vs. Hesitant**: Two different weight configurations both achieve $100\%$ accuracy on OR3. One outputs $0.99$ and $0.01$, while the other outputs $0.60$ and $0.40$. Which is better, and why does confidence matter once there is a loss function?
4. **Hypercube geometry**: Sketch the 3D unit cube of 8 inputs. Can you draw a single flat plane that cuts off $(1, 1, 1)$ for AND3? Why is it impossible for XOR3?
5. **Defensive programming**: Why does `weighted_sum` explicitly raise `ValueError` on mismatched input dimensions rather than quietly letting Python's `zip()` truncate?
6. **Flashcard check**:
   - For $\mathbf{w} = [2, 2]$ and $b = -3$, which of the inputs $(0,0), (0,1), (1,0), (1,1)$ produce an output above $0.5$? *(Only $(1,1)$: weighted sums are $-3, -1, -1, +1$.)*
   - If you double every weight and the bias, do the predictions change? Does the confidence change? *(Predictions stay identical because the zero-crossing is unchanged; confidence increases as activations move closer to $0$ and $1$.)*

---

## Common mistakes & troubleshooting

| Test failure / symptom | Likely cause | Fix |
|---|---|---|
| `OverflowError` in sigmoid tests | Used textbook $1 / (1 + e^{-z})$ for large negative $z$ | Use dual branch: $e^z / (1 + e^z)$ when $z < 0$ |
| Reproducibility test fails (`a != b`) | Created a local generator inside `random()` | Use the passed-in `rng.uniform(-1, 1)` |
| `type(w) is float` fails | Stored NumPy scalar floats | Wrap each drawn number in Python's `float(...)` |
| `weighted_sum` bias test fails | Started accumulation at `0.0` instead of `self.b` | Initialize accumulator `z = self.b` |
| `test_your_neurons_are_confident` fails on OR/AND | Weights are too small (e.g. $1.0$ instead of $10.0$) | Scale weights up to $10.0$ to push activations into saturation ($> 0.9$ or $< 0.1$) |

---

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1:** Branch on `z >= 0`. For negative $z$, use `e = math.exp(z)` and return `e / (1.0 + e)`.
- **Task 2:** `[float(rng.uniform(-1, 1)) for _ in range(d_in)]`.
- **Task 3:**
  - Check `len(x) != len(self.w)` first.
  - Start with `z = self.b`, then loop over `zip(self.w, x)` adding `w_i * x_i`.
- **Task 4:** `sum(1 for e in data if neuron.predict(e.x) == e.y) / len(data)`.
- **Task 5:**
  - For OR: weights $[10.0, 10.0, 10.0]$ with bias $-5.0$.
  - For AND: weights $[10.0, 10.0, 10.0]$ with bias $-25.0$.
</details>


