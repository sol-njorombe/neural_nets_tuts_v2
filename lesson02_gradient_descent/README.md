# Lesson 2 — Loss and gradient descent

**Read [THEORY.md](THEORY.md) first.**

## What you build

A loss function, the exact analytical gradient of that loss for every weight and
the bias, and a training loop that steps downhill. Your neuron learns OR, AND, and
FIRST_INPUT automatically from data, and hits the XOR wall in a way you will be
able to explain geometrically.

## Setup

Run the test suite as you work through each task:

```bash
pytest lesson02_gradient_descent -x              # stop on first failure
pytest lesson02_gradient_descent -k gradients   # run only gradient tests
python lesson02_gradient_descent/descent.py      # after task 4: run the demo
```

`Neuron`, `Example`, `accuracy` and the four datasets come from Lesson 1, already
finished, in `nnlib.scalar`. If `import nnlib` fails, run `pip install -e .`
(or `uv sync`) from the repository root.

---

## Tasks

Work through `descent.py` in order:

| # | Function | What it does | Theory |
|---|---|---|---|
| 1 | `squared_error`, `loss` | Compute $(a - y)^2$ and the dataset mean squared error $L$ | [THEORY §1](THEORY.md#1-a-score-for-being-wrong-the-mean-squared-error) |
| — | `train_by_wiggling` | Read the given coordinate search to see the baseline | [THEORY §2](THEORY.md#2-wiggling-to-learn-the-naive-baseline) |
| 2 | `gradients` | Calculate exact analytical $\frac{\partial L}{\partial \mathbf{w}}$ and $\frac{\partial L}{\partial b}$ in one pass | [THEORY §3, §4](THEORY.md#3-the-gradient-calculus-does-all-weights-at-once) |
| 3 | `step` | Update weights and bias in place against the gradient | [THEORY §3, §5](THEORY.md#5-the-step-size-learning-rate) |
| 4 | `train` | Run the training loop for $N$ epochs and return the loss history | [THEORY §5](THEORY.md#5-the-step-size-learning-rate) |

### Task 1: Measuring wrongness (`squared_error` and `loss`)
- `squared_error(a, y)` returns $(a - y)^2$ for a single prediction.
- `loss(neuron, data)` computes the mean squared error across all examples in `data`.
  Remember that $L$ is an **average** (divide by $n = \text{len}(\text{data})$), not a sum.
  A clueless neuron ($a = 0.5$) must produce $L = 0.25$.

### Given: Trial and error (`train_by_wiggling`)
- Read the implementation in `descent.py`.
- It nudges each weight $+h$ and $-h$, re-evaluating the loss across all examples each time.
- Observe how many loss evaluations it consumes to train a simple gate.

### Task 2: The analytical gradient (`gradients`)
- Calculate $\delta = 2(a - y) \cdot a(1 - a)$ for each example, where $a = \text{neuron.forward}(e.x)$.
- Accumulate the weight gradients: $\frac{\partial L_k}{\partial w_i} = \delta \cdot x_i$.
- Accumulate the bias gradient: $\frac{\partial L_k}{\partial b} = \delta$.
- Divide the accumulated sums by $n = \text{len}(\text{data})$ to return the dataset mean gradient.
- Return a `Gradients(dw=dw, db=db)` dataclass.
- **Do not modify the neuron in `gradients()`.**
- The test suite compares your result against `numerical_gradients()` using gradient checking ($\text{tolerance} < 10^{-7}$).

### Task 3: Taking a downhill step (`step`)
- Update every weight in-place: $w_i \leftarrow w_i - \eta \cdot \text{grads.dw}[i]$.
- Update the bias in-place: $b \leftarrow b - \eta \cdot \text{grads.db}$.
- Notice the **minus sign**: moving against the gradient moves downhill toward lower loss.

### Task 4: The training loop (`train`)
- For each epoch in `range(epochs)`:
  1. Record the current loss in `history` **before** taking the step (so `history[0]` is the untrained baseline).
  2. Compute exact gradients: `grads = gradients(neuron, data)`.
  3. Update parameters: `step(neuron, grads, lr)`.
- Return `history: list[float]`.

---

## Running the demo

Once Task 4 is complete, run:

```bash
python lesson02_gradient_descent/descent.py
```

Expected output:

```text
OR3          loss 0.407 -> 0.0022   accuracy 1.00
AND3         loss 0.217 -> 0.0068   accuracy 1.00
FIRST_INPUT  loss 0.250 -> 0.0007   accuracy 1.00
XOR3         loss 0.282 -> 0.2500   accuracy 0.75
XOR3 is stuck near 0.25 however long you train. THEORY §6 explains why.
```

---

## Experiments

1. **Learning rate exploration:**
   In `descent.py`, train on OR3 with `lr = 0.001`, `0.1`, `1.0`, `10.0`, and `100.0`.
   Plot or print the loss every 200 epochs. At what point does steady convergence turn into oscillation or divergence?
2. **Feature selection (Credit Assignment):**
   Print the trained neuron for `FIRST_INPUT`. What happened to $w_1$ and $w_2$?
   Why did the gradient descent algorithm automatically drive irrelevant weights toward zero?
3. **Wiggling versus descent efficiency:**
   Compare the cost of wiggling versus analytical gradient descent using:
   ```python
   from nnlib.scalar import truth_table
   data = truth_table(lambda x: x[0] == 1, d_in=10)
   ```
   Count how many full passes over the dataset each method requires to reach $L < 0.01$.
4. **The XOR wall:**
   Train on `XOR3` across 5 different random seeds. Does any run escape the $0.25$ loss trap?
   Review the 3D cube diagram in THEORY §6 to explain why a single hyperplane cannot separate alternating vertices.

---

## The limit

Two fundamental limits remain:

1. **XOR cannot be solved by a single neuron.**
   Regardless of training time, a single linear threshold cannot compute non-linearly
   separable functions. We need multiple neurons organized into **layers**.
2. **Python loops do not scale.**
   `gradients()` loops over every example, and inside that over every input weight.
   A modest network with 100 neurons processing 10,000 examples with 100 inputs would
   require over 100 million slow Python operations per epoch.

In **Lesson 3**, we introduce NumPy arrays, broadcasting, and matrix shapes to eliminate
Python loops. In **Lesson 4**, we vectorize a single neuron. In **Lesson 5**, we add
hidden layers and backpropagation to conquer XOR.

---

## Self-check & discussion questions

Test your mental model:
1. **Squared vs. Absolute Error**: Why do we square the error $(a - y)^2$ rather than taking its absolute value $|a - y|$? *(Both penalize errors, but squaring is smooth and differentiable everywhere with a continuous linear derivative $2(a - y)$, and punishes large misses quadratically.)*
2. **Input Blame**: Why is each weight's slope $\frac{\partial L}{\partial w_i}$ equal to $\delta$ multiplied by its input feature $x_i$? What does an input of $0.0$ mean for its weight's gradient?
3. **Loss curve diagnostics**: Suppose your loss history across epochs reads: $0.25, 0.20, 0.31, 0.18, 0.40 \dots$. What is happening to the optimization, and what hyperparameter should you adjust? *(The learning rate $\eta$ is too large, causing the step to overshoot the valley floor and bounce across the walls; reduce $\eta$.)*
4. **The XOR ceiling**: The loss on `XOR3` gets stuck near $0.25$ and never drops. Is gradient descent failing or buggy? How would you prove to a colleague that gradient descent is functioning correctly?
5. **Wiggling versus calculus**: Wiggling was competitive on 3 inputs. Why does nobody train real neural networks using wiggling? *(Because wiggling evaluates $2(K)$ passes per step, which takes millions of passes per step on modern networks, whereas analytical gradients compute slopes for all parameters in a single pass.)*
6. **Flashcard check**:
   - An example has $\delta = -0.3$ and input features $(1.0, 0.0, 1.0)$. What are its raw contributions to $\frac{\partial L}{\partial \mathbf{w}}$ before dataset averaging? *$(-0.3, 0.0, -0.3)$.*
   - If a weight's slope is $+2.0$ and $\eta = 0.1$, what happens to the weight in `step()`? *(It falls by $0.2$.)*

---

## Common mistakes & troubleshooting

| Test failure / symptom | Likely cause | Fix |
|---|---|---|
| Gradient check fails by a factor of $n$ | Summed per-example gradients without dividing by $n$ | Divide accumulated gradient sums by $\text{len}(\text{data})$ |
| Bias gradient wrong, weights right | Multiplied the bias term by an input | $\frac{\partial z}{\partial b} = 1$, so bias gradient is $\delta$ alone without multiplying by $x_i$ |
| Loss rises during training | Added the gradient instead of subtracting | In `step()`, subtract: $\theta \leftarrow \theta - \text{lr} \cdot \text{slope}$ |
| `history[0]` test fails | Recorded loss after taking the step | Call `history.append(loss(neuron, data))` **before** calling `step()` |
| XOR test "fails" because it solved XOR | Passed wrong dataset or changed architecture | A single linear neuron cannot solve XOR; verify you passed `XOR3` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **loss:** `return sum(squared_error(neuron.forward(e.x), e.y) for e in data) / len(data)`
- **gradients:** Initialize `dw = [0.0] * len(neuron.w)` and `db = 0.0`.
  For each example `e` in `data`:
  ```python
  a = neuron.forward(e.x)
  delta = 2 * (a - e.y) * a * (1 - a)
  for i, x_i in enumerate(e.x):
      dw[i] += delta * x_i / n
  db += delta / n
  ```
- **Common debugging checks:**
  - If the gradient check fails by a factor of 8, you forgot to divide by $n = \text{len}(\text{data})$.
  - If the sign is inverted, check that you wrote `(a - e.y)` and not `(e.y - a)`.
  - In `step`, ensure you subtract: `neuron.w[i] -= lr * grads.dw[i]`.
  - In `train`, record `history.append(loss(neuron, data))` **before** calling `step()`.
</details>


