# Lesson 6 — Theory: layers, softmax, minibatches

By the end of this lesson your own code reads handwritten digits at about 97%
accuracy. Three ideas get you there: layers as interchangeable bricks, a loss
for choosing between ten classes, and training on small random batches.

## 1. The data

1,797 images of handwritten digits, each 8×8 pixels. Each image is flattened to
64 numbers between 0 (white) and 1 (dark), and labelled with the digit it shows.

```
label: 5          X row: 64 numbers         y: 5
  --@@@@@@%%++
  ++@@##==--::
  **@@
  ++@@--
  ..%%%%..
    ..@@##
    ::@@++
  ++@@**
```

`load_digits()` returns a `Digits` dataclass: 1,437 training images and 360
**validation** images. You never train on the validation images; they are how
you find out whether the network learned digits or memorised its homework
(lesson 7 is about exactly that).

## 2. Layers are bricks

Lesson 5's `backward` was hand-written for one architecture. But §4 of that
lesson showed that every layer does the same job. Make that job an interface:

```
forward(X)   →  output                      and remember what backward needs
backward(dY) →  dX = ∂L/∂(input)            and store ∂L/∂(own parameters)
```

`dY` is "how the loss changes as my output changes". A layer's `backward` is the
chain rule for its own small step and nothing more. It does not know what comes
before or after it. That is what makes layers stackable.

**`Dense`** is lesson 5's `W1, b1`. Forward saves its input; backward is the table
from lesson 3 §8:

| Forward | Backward | Shapes |
|---|---|---|
| Y = X @ W + b | dW = X.T @ dY | (d_in, d_out), like W |
| | db = dY.sum(axis=0) | (d_out,), like b |
| | dX = dY @ W.T | (n, d_in), like X: returned to the previous layer |

**Activations become bricks too.** Lesson 5 multiplied by `H(1 − H)` inside
`backward`. Now the sigmoid is its own layer whose backward multiplies by its
own slope. That means you can swap activations by changing one item in a list.

## 3. ReLU

```
ReLU(x) = max(0, x)          slope: 1 where x > 0,  0 where x < 0
```

ReLU is the modern default for hidden layers, for a reason you met in lesson 0
§5 and lesson 5 §5: the sigmoid's slope is at most 0.25, so blame passing back
through several sigmoids shrinks by at least 4× per layer. Through ten layers
that is a factor of a million: the early layers barely learn. This is the
**vanishing gradient** problem. ReLU's slope is exactly 1 wherever the neuron is
active, so blame passes back undiminished.

Its backward pass is a mask: keep the blame where the input was positive, zero it
elsewhere.

| Math | Shape | Code |
|---|---|---|
| A = max(0, X) | like X | `mask = X > 0; A = X * mask` |
| dX = dY ⊙ [X > 0] | like X | `dX = dY * mask` |

## 4. Ten answers instead of one

A digit classifier needs a score for each of the ten classes. The last `Dense`
layer has 10 outputs, called **logits**: raw scores, any real number.

**Softmax** turns each row of scores into probabilities:

```
pⱼ = e^(zⱼ) / Σₖ e^(zₖ)          positive, and each row sums to 1
```

Bigger score, bigger probability, and the exponential exaggerates the gaps.
Subtracting the row's largest score before exponentiating gives the same answer
(it cancels top and bottom) and avoids `e^1000 = inf`.

**Cross-entropy** is the loss: minus the log of the probability the model gave
to the **correct** class.

```
L = −(1/n) Σᵣ log P[r, yᵣ]
```

It works like a bet: say 99% and be right, lose 0.01. Say 99% on the wrong class
(so 1% on the right one), lose 4.6. Confident mistakes are punished hard.

**The combined backward is simple.** Differentiating cross-entropy through
softmax (a few lines of chain rule, worth doing once on paper) gives:

```
∂L/∂logits = (P − one_hot(y)) / n
```

"For each class, how far was my probability from the truth?" The correct
class's entry is negative (raise it); every other entry is positive (lower them).

**Worked example.** Two classes, logits `[0, ln 3]`, correct class 1.

| | Class 0 | Class 1 |
|---|---|---|
| e^z | 1 | 3 |
| P | 0.25 | 0.75 |
| one-hot | 0 | 1 |
| P − one-hot | +0.25 | −0.25 |

Loss = −ln 0.75 = 0.288.

## 5. A network is a list

```python
model = Sequential([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)])
```

`Sequential.forward` calls each layer in order; `Sequential.backward` calls each
in reverse, passing the returned blame along. `params()` collects every
`(parameter, gradient)` pair, so one `sgd_step` updates the whole model.

Adding a layer is now editing a list. In lesson 5 it was a rewrite.

**In place matters.** `params()` hands out references to the layer's arrays.
`p -= lr * g` changes that array; `p = p - lr * g` makes a new array and leaves
the layer's weights untouched. The same goes for filling `dW` in `backward`.

## 6. Minibatches

Lesson 4 used every example for every step (accurate, but one step per pass).
Lesson 2's online version used one at a time (many steps, but noisy). The
standard compromise is **minibatches**:

1. Shuffle the training rows.
2. Cut them into batches of about 32.
3. One forward, backward and update per batch.

Like polling 32 voters rather than one or the whole country: a good enough
estimate of the slope, 45 times per pass over these 1,437 images. This is
**stochastic gradient descent (SGD)**. Shuffling every epoch means the batches
differ each time, which also helps training escape poor valleys.

| Math | Shape | Code |
|---|---|---|
| logits = model(X_batch) | (batch, 10) | `model.forward(X_batch)` |
| P = softmax(logits) | (batch, 10) | `softmax(...)` |
| L = cross-entropy | one number | `cross_entropy(P, y_batch)` |
| ∂L/∂logits | (batch, 10) | `cross_entropy_backward(P, y_batch)` |
