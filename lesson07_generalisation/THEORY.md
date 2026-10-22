# Lesson 7 — Theory: generalisation, initialisation, optimisers

Lesson 6 trained a network that reads digits. This lesson is about everything
that decides whether training works: measuring the right thing, starting from
sensible weights, choosing how to step, knowing when to stop, and reading the
curves when something goes wrong.

## 1. Generalisation: the only score that counts

A network is useful only if it works on examples it has **never seen**. Training
accuracy measures something else: how well it fits the examples it was shown.

**Analogy.** A student given last year's exam to practise on can score 100% by
memorising the answers, and learn nothing about the subject. To find out what
they actually learned, you set questions they have not seen.

So the data is split before training starts:

| Set | Used for | Rule |
|---|---|---|
| Training | Computing gradients | The only data the network learns from |
| Validation | Choosing settings (learning rate, size, when to stop) | Never used for gradients |
| Test (optional) | One final, honest score | Looked at once, at the very end |

Rows must be **shuffled before splitting** (the data may be sorted), each row
must land in exactly one set, and rows that are near-copies of each other must
land in the same set. If near-identical rows end up on both sides, the
validation score is inflated; this is called **leakage**. A real example: photos
from the same shoot look alike and are edited alike, so a photo-editing model
must be split by shoot, not by photo.

## 2. Underfitting and overfitting

Watch two numbers per epoch, training accuracy and validation accuracy:

```
accuracy                                    accuracy
  1.0 ┤        ___________ train              1.0 ┤
      │      _/                                   │
      │    _/  ‾‾‾‾‾‾‾‾‾‾‾ val                    │     _____ train
      │   /   /                                   │   _/ ____ val
      │  /  _/       ← gap                        │  / _/
      │ / _/                                      │ /_/
      └──────────────── epochs                    └──────────────── epochs
         overfitting: memorising                     underfitting: not learning
```

| Pattern | Name | Typical causes | Typical fixes |
|---|---|---|---|
| Train high, validation much lower | **Overfitting** | Too little data for the model's size; trained too long | More data, smaller model, early stopping, regularisation |
| Train low, validation similar | **Underfitting** | Model too small, learning rate too low, too few epochs | Bigger model, higher learning rate, train longer |
| Train loss rising, or nan | **Diverging** | Learning rate too high; bad initialisation | Lower learning rate; fix initialisation |

Running `python lesson07_generalisation/generalisation.py` trains a 256-neuron network on just 60 images.
It reaches 100% training accuracy within 50 epochs while validation stays near
79%: a gap of over 20 points, the textbook picture of memorisation.

## 3. Initialisation: start at a sensible scale

Lessons 1–5 drew weights from `uniform(−1, 1)`. With 3 inputs that was fine.
With 100 inputs it is not.

**Why.** A neuron's weighted sum adds up `d_in` terms. If inputs have variance
about 1 and weights have variance `σ²`, the sum has variance about `d_in · σ²`.
A ReLU then zeroes roughly half the values, halving that. So each layer
multiplies the signal's variance by about:

```
gain = d_in · σ² / 2
```

| Initialisation | σ² | Gain per layer at d_in = 100 | After 20 layers |
|---|---|---|---|
| uniform(−1, 1) | 1/3 | ≈ 17 | explodes (≈ 10¹¹) |
| normal(0, 0.01) | 0.0001 | ≈ 0.005 | vanishes (≈ 10⁻²³) |
| **He: normal(0, √(2/d_in))** | 2 / d_in | **1** | stays ≈ 1 |

He initialisation chooses σ² so the gain is exactly 1. Signals neither explode
nor vanish, and neither do the gradients flowing back. That is why lesson 6's
`Dense` used it. `signal_scales` measures this directly.

| Math | Shape | Code |
|---|---|---|
| W ~ Normal(0, √(2/d_in)) | (d_in, d_out) | `rng.normal(0.0, math.sqrt(2 / d_in), (d_in, d_out))` |

## 4. The learning rate

From lesson 2 §5: too small crawls, too large overshoots. In practice:

- Try values a factor of 10 apart (0.001, 0.01, 0.1, 1) and compare validation
  loss after a few epochs.
- A training loss that is flat from the first epoch usually means too small, or
  dead ReLUs from a step that was far too large.
- A training loss that rises or turns into `nan` means too large.

## 5. Better ways to step

Plain SGD uses one learning rate for every parameter. That hurts when the loss
surface is a **long narrow valley**: steep across, gentle along.

```
   steep direction ↕     ╲   ╱  ╲   ╱  ╲   ╱      SGD: zig-zags across the valley,
                          ╲ ╱    ╲ ╱    ╲ ╱       barely moves along it
   gentle direction →  ───────────────────────▶
```

The learning rate must be small enough not to overshoot the steep direction, so
progress in the gentle direction is painfully slow. The tests use
`f(p) = ½(100 p₀² + p₁²)`, which is 100× steeper in one direction.

**Momentum: a heavy ball.** Keep a running velocity and step along it:

```
v ← β·v + g          p ← p − lr·v          (β ≈ 0.9)
```

Zig-zags across the valley cancel out in `v`; consistent pushes along it add
up. With β = 0.9, a steady gradient builds a velocity of about 10× its size.

**Adam: a step size per parameter.** Keep running averages of the gradient (`m`)
and of its square (`v`), and divide:

```
m ← β₁·m + (1 − β₁)·g             running mean of g        (β₁ = 0.9)
v ← β₂·v + (1 − β₂)·g²            running mean of g²       (β₂ = 0.999)
m̂ = m / (1 − β₁ᵗ)    v̂ = v / (1 − β₂ᵗ)                     bias correction
p ← p − lr · m̂ / (√v̂ + ε)
```

`m̂ / √v̂` is roughly "the gradient divided by its typical size", so every
parameter moves by about `lr` per step whatever the scale of its gradient. Steep
and gentle directions get similar step sizes. The bias correction fixes the fact
that `m` and `v` start at zero: without it the first steps would be far too small.

Adam is the default choice for most networks today, with `lr` around 0.001.

| Optimiser | State per parameter | Update |
|---|---|---|
| SGD | none | p −= lr·g |
| Momentum | v | v = βv + g; p −= lr·v |
| Adam | m, v, and step count t | p −= lr·m̂/(√v̂ + ε) |

## 6. Early stopping

When validation loss stops improving, more training only memorises. **Early
stopping** watches validation loss each epoch, remembers the weights from the
best epoch, stops after `patience` epochs without improvement, and restores the
best weights at the end.

It needs **copies** of the parameters (`p.copy()`): saving `p` itself saves a
reference to an array that training keeps changing.

## 7. Reading the curves

`diagnose` turns §2's table into code, checked in order:

1. **Diverging:** final training loss is `nan`/`inf`, or higher than at the start.
2. **Underfitting:** final training accuracy below 0.9.
3. **Overfitting:** training accuracy beats validation by more than 0.1.
4. Otherwise **healthy**.

The thresholds are judgement calls for this dataset, not universal constants.
What generalises is the order: first check that training works at all, then
that it fits the training data, and only then compare with validation.

## 8. Other tools you will meet

- **L2 regularisation (weight decay):** add `λ·Σw²` to the loss, which pulls
  weights towards zero and discourages memorising. Its gradient is just `2λw`.
- **Dropout:** during training, randomly zero a fraction of activations each
  step, so no neuron can rely on a particular other neuron.
- **Data augmentation:** make more training data by shifting, flipping or
  recolouring existing examples.
- **Learning-rate schedules:** start larger and decay over training.

PyTorch provides all of these as one-liners (lesson 9). After this lesson, you
know what each one is for.
