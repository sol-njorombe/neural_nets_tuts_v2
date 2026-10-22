# Lesson 7 — Theory: generalisation, initialisation, optimisers

Lesson 6 trained a network that reads digits. This lesson is about everything
that decides whether training works: measuring the right thing, starting from
sensible weights, choosing how to step, knowing when to stop, and reading the
curves when something goes wrong.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Formulate Data Leakage & Splitting**: Explain why validation splits must be shuffled and strictly partitioned before model exposure, identifying real-world leakage vectors.
2. **Diagnose Training Dynamics**: Classify training trajectories into *Healthy*, *Overfitting*, *Underfitting*, or *Diverging* from loss and accuracy curves.
3. **Derive He (Kaiming) Initialization**: Derive $\text{std} = \sqrt{2/d_{\text{in}}}$ by tracking signal variance through linear layers and half-rectified ReLU activations.
4. **Contrast First- and Second-Moment Optimizers**: Compare vanilla SGD, Momentum, and Adam, explaining how running averages dampen cross-valley oscillations and why bias correction is necessary.
5. **Implement Early Stopping**: Implement state checkpointing using explicit array cloning (`copy()`) to prevent overfitting.

---

## 1. Generalisation: the only score that counts

A neural network is valuable only if it generalizes to unseen test instances. Training accuracy measures something else: how faithfully it fits the examples it was shown.

### The Exam Analogy
A student given last year's exam to practice on can score 100% simply by memorising the exact questions and answers without learning the underlying principles. To evaluate actual mastery, you must test on previously unseen questions.

Data is therefore partitioned before any training commences:

| Partition | Role | Golden Rule |
|---|---|---|
| **Training Set** | Parameter optimization ($\nabla_\theta L$) | The only data that computes gradients and updates weights |
| **Validation Set** | Hyperparameter tuning ($lr$, architecture, patience) | Evaluated periodically; never updates weights |
| **Test Set** | Final unbiased performance assessment | Evaluated once at the very end of research |

### Data Leakage
Rows must be **shuffled with a random permutation before splitting** (datasets are frequently sorted by class or timestamp). Every example must land in exactly one partition.

Crucially, correlated or duplicate examples must not cross partition boundaries:
- In clinical prediction: all scans from the same patient must reside in the same split.
- In photography / vision: all frames from the same photoshoot or video clip must stay together.
If identical or near-identical examples appear in both training and validation, validation accuracy is artificially inflated; this failure is called **data leakage**.

---

## 2. Underfitting and Overfitting

By tracking training and validation curves side-by-side, you diagnose model health:

```
accuracy                                    accuracy
  1.0 ┤        ___________ train              1.0 ┤
      │      _/                                   │
      │    _/  ‾‾‾‾‾‾‾‾‾‾‾ val                    │     _____ train
      │   /   /                                   │   _/ ____ val
      │  /  _/       ← generalization gap         │  / _/
      │ / _/                                      │ /_/
      └──────────────── epochs                    └──────────────── epochs
         Overfitting (memorisation)                  Underfitting (capacity deficit)
```

| Trajectory Pattern | Diagnostic Label | Primary Root Causes | Prescribed Solutions |
|---|---|---|---|
| Train acc $> 0.9$, Val acc $\ge 0.10$ lower | **Overfitting** | Model capacity exceeds dataset size; over-trained | More training data, smaller model, L2 regularization, dropout, early stopping |
| Train acc $< 0.9$, Val acc similar | **Underfitting** | Insufficient model capacity, learning rate too low, dead neurons | Wider/deeper network, increase $lr$, train longer, switch to Adam |
| Loss explodes to `nan`/`inf` or increases | **Diverging** | Learning rate too high, weight variance exploding | Lower $lr$ by $10\times$, check gradient clipping, verify He initialization |

Running `python lesson07_generalisation/generalisation.py` trains a 256-neuron network on just 60 images.
It reaches 100% training accuracy within 50 epochs while validation stays near
79%: a gap of over 20 points, the classic picture of memorisation.

---

## 3. Initialisation: Scaling Signal Variance

Lessons 1–5 drew weights from $\text{Uniform}(-1, 1)$. With $d_{\text{in}} = 3$ this worked. With $d_{\text{in}} = 100$ or $1000$, it fails catastrophically.

### Mathematical Derivation of He (Kaiming) Initialization

Consider a single neuron computing $z = \sum_{i=1}^{d_{\text{in}}} w_i x_i$.
Assume inputs $x_i$ and weights $w_i$ are independent, zero-mean random variables with variances $\text{Var}(x)$ and $\sigma^2 = \text{Var}(w)$.
By the variance of independent products:

$$\text{Var}(z) = \sum_{i=1}^{d_{\text{in}}} \text{Var}(w_i x_i) = d_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x) = d_{\text{in}} \sigma^2 \text{Var}(x)$$

Next, pass $z$ through a **ReLU** activation:
$$\text{ReLU}(z) = \max(0, z)$$
Because $z$ is symmetric around zero, exactly half of the distribution is truncated to zero.
Assuming $z$ is normally distributed:

$$\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \text{Var}(z) = \frac{1}{2} d_{\text{in}} \sigma^2 \text{Var}(x)$$

To preserve activation variance from layer to layer ($\text{Var}(\text{output}) = \text{Var}(x)$), the gain must equal $1$:

$$\text{Gain} = \frac{1}{2} d_{\text{in}} \sigma^2 = 1 \implies \sigma^2 = \frac{2}{d_{\text{in}}} \implies \sigma = \sqrt{\frac{2}{d_{\text{in}}}}$$

| Initialization Strategy | Weight Variance $\sigma^2$ | Gain per Layer ($d_{\text{in}} = 100$) | Signal after 20 Layers |
|---|---|---|---|
| `Uniform(-1, 1)` | $1/3 \approx 0.333$ | $\approx 16.7$ | Explodes ($16.7^{20} \approx 10^{24}$) |
| `Normal(0, 0.01)` | $0.0001$ | $0.005$ | Vanishes ($0.005^{20} \approx 10^{-46}$) |
| **He Normal: $\mathcal{N}(0, \sqrt{2/d_{\text{in}}})$** | $2 / d_{\text{in}}$ | **$1.0$** | **Stable ($\approx 1.0$)** |

```python
# In code:
W = rng.normal(0.0, math.sqrt(2.0 / d_in), size=(d_in, d_out))
```

---

## 4. The Learning Rate

Learning rate is the single most critical hyperparameter:
- Values that are a factor of 10 apart ($0.001, 0.01, 0.1, 1.0$) should be evaluated systematically.
- Flat loss from epoch 1 indicates either an excessively small $lr$ or dead ReLUs caused by an oversized first step.
- Escalating loss or `nan` indicates numerical divergence from an oversized $lr$.

---

## 5. Better Ways to Step: Optimizers

Plain SGD applies a uniform scalar learning rate across every parameter:
$$p \leftarrow p - \text{lr} \cdot g$$
When the loss landscape resembles an anisotropic "ravine" (steep walls along dimension 0, gentle slope along dimension 1), SGD oscillates violently across the ravine while making minuscule progress along the base.

```
   steep direction ↕     ╲   ╱  ╲   ╱  ╲   ╱      SGD: zig-zags across the valley,
                          ╲ ╱    ╲ ╱    ╲ ╱       barely moves along it
   gentle direction →  ───────────────────────▶
```

### 1. SGD with Momentum (The Heavy Ball)
Maintains an exponential velocity vector $v$:

$$v \leftarrow \beta v + g$$
$$p \leftarrow p - \text{lr} \cdot v \qquad (\beta \approx 0.9)$$

Orthogonal oscillations cancel out through signed summation, whereas consistent gradient pushes along the ravine accumulate, accelerating descent by up to $\frac{1}{1 - \beta} = 10\times$.

### 2. Adam (Adaptive Moment Estimation)
Maintains running estimates of both first moments (mean gradient $m$) and second uncentered moments (mean squared gradient $v$):

$$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad (\beta_1 = 0.9)$$
$$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad (\beta_2 = 0.999)$$

#### Bias Correction
Because $m_0 = 0$ and $v_0 = 0$, uncorrected running averages are heavily biased toward zero in initial epochs:
$$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = \mathbb{E}[g] (1 - \beta_1^t)$$
Dividing by $(1 - \beta^t)$ eliminates this initial transient deficit:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

#### Parameter Step
$$p \leftarrow p - \text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Because $\frac{\hat{m}}{\sqrt{\hat{v}}} \approx \pm 1$, every parameter moves approximately by magnitude $\text{lr}$ regardless of gradient scale, equalizing step sizes across steep and flat dimensions.

---

## 6. Early Stopping

When validation loss ceases to decrease, continuing to train only memorises noise.
**Early stopping**:
1. Monitors validation loss at each epoch.
2. Clones parameter arrays whenever a new minimum validation loss is achieved:
   `self.best_params = [p.copy() for p, _ in model.params()]`
3. Halts training when validation loss fails to improve for `patience` consecutive epochs.
4. Restores best weights in place:
   `for (p, _), best in zip(model.params(), self.best_params): p[...] = best`

---

## 7. Automated Run Diagnosis

`diagnose(history)` evaluates training health through a strict prioritized decision tree:

```
[Is final loss NaN/inf or higher than epoch 0?]
   ├── YES ──▶ Diverging
   └── NO  ──▶ [Is final train accuracy < 0.90?]
                 ├── YES ──▶ Underfitting
                 └── NO  ──▶ [Is (train_acc - val_acc) >= 0.10?]
                               ├── YES ──▶ Overfitting
                               └── NO  ──▶ Healthy
```
