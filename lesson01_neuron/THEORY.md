# Lesson 1 — Theory: one neuron

## 1. Concept: a neuron is a mixing desk

A sound engineer has one channel per instrument and a slider on each channel.
The slider sets how much of that instrument reaches the speakers. A master
level shifts everything up or down. A limiter on the output stops it going
beyond a fixed range.

A neuron is that desk:

| Mixing desk | Neuron | Symbol |
|---|---|---|
| Instruments | Inputs | x₀, x₁, x₂ |
| Sliders | **Weights**: how much each input counts. Negative means it counts *against*. | w₀, w₁, w₂ |
| Master level | **Bias**: a head start, or a hurdle, applied whatever the inputs | b |
| Mixed signal | **Weighted sum** | z |
| Limiter | **Activation**: squashes z into a fixed range | a = σ(z) |

```
   x₀ ──(w₀)──┐
   x₁ ──(w₁)──┼──▶  z = Σ wᵢxᵢ + b  ──▶  a = σ(z)  ──▶  output, between 0 and 1
   x₂ ──(w₂)──┘          ▲
                         b
```

The entire "knowledge" of a neural network is the values of its weights and
biases. There is nothing else in it. Learning means changing those numbers.

## 2. The forward pass

Running inputs through to an output is the **forward pass**:

```
z = w₀x₀ + w₁x₁ + w₂x₂ + b
a = σ(z)
```

**Worked example.** `w = [1, −2, 0.5]`, `b = 0.5`, `x = (1, 0, 1)`.

| Step | Calculation | Value |
|---|---|---|
| w₀x₀ | 1 × 1 | 1 |
| w₁x₁ | −2 × 0 | 0 — an input of 0 silences its weight entirely |
| w₂x₂ | 0.5 × 1 | 0.5 |
| z | 1 + 0 + 0.5 + 0.5 | 2 |
| a | σ(2) | 0.8808 |

**Three ways.**

| Math | Shape | Code |
|---|---|---|
| z = Σᵢ wᵢxᵢ + b | `w` and `x` both length d_in; z one number | `neuron.weighted_sum(x)` |
| a = σ(z) | one number | `neuron.forward(x)` |
| ŷ = 1 if a > 0.5 else 0 | one integer | `neuron.predict(x)` |

## 3. The sigmoid

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

| z | −6 | −2 | 0 | 2 | 6 |
|---|---|---|---|---|---|
| σ(z) | 0.002 | 0.119 | 0.5 | 0.881 | 0.998 |

Read the output as "how sure the neuron is that the answer is 1". Large positive
`z`: nearly certain yes. Large negative: nearly certain no. Exactly 0: no idea.

Why not just output 1 if `z > 0`, else 0? Because a hard step has a derivative
of zero almost everywhere (lesson 0: flat means "turning does nothing"). Lesson 2
learns by following slopes, so the activation must have slopes to follow. The
sigmoid is a smooth step.

**A numerical trap.** `math.exp(1000)` overflows. For `z = −1000` the textbook
formula computes `e^1000` and crashes. Multiply the top and bottom of the
formula by `eᶻ`:

```
1 / (1 + e⁻ᶻ)   =   eᶻ / (eᶻ + 1)
```

Same value, but for negative `z` it computes `eᶻ`, which is tiny instead of
huge. Use the first form when `z ≥ 0` and the second when `z < 0`. Real
libraries do exactly this.

## 4. What a single neuron can compute

The neuron says 1 when `z > 0`, that is, when `w₀x₀ + w₁x₁ + w₂x₂ > −b`. So the
bias works as a **threshold** the weighted sum has to clear.

**OR3** ("at least one input on") with all weights equal to 10:

| Inputs on | Weighted sum | Need it to be |
|---|---|---|
| 0 | 0 | below the threshold |
| 1 | 10 | above |
| 2 | 20 | above |
| 3 | 30 | above |

Any threshold between 0 and 10 works. `b = −5` puts it at 5.

AND3 is the same idea with a higher hurdle. Work out the bias yourself in task 5.

**Why large weights?** With weights of 10, `z` sits at least 5 away from zero
for every row, so `σ(z)` is near 0 or near 1. Weights of 1 with bias −0.5 also
put every row on the correct side, but give outputs like 0.62: right, but
hesitant. Confidence will matter once there is a loss to minimise.

**Geometric picture.** The 8 possible inputs are the corners of a cube. The
equation `w·x + b = 0` is a flat plane through that cube. The neuron says 1 on
one side of the plane and 0 on the other. OR and AND can each be separated by
one flat cut. Lesson 2 shows that XOR cannot, which is the whole reason
networks have more than one layer.

## 5. Why a typed class?

The neuron is a `dataclass`:

```python
@dataclass
class Neuron:
    w: list[float]  # one weight per input, length d_in
    b: float = 0.0
```

- **The editor knows what `neuron.w` is.** You never have to remember whether
  it was `neuron["weights"]` or `neuron["w"]`.
- **The methods carry the math.** `neuron.forward(x)` reads like the formula.
- **Mistakes fail loudly.** `zip(w, x)` silently stops at the shorter list, so a
  neuron with 3 weights given 2 inputs would quietly ignore its third weight.
  `weighted_sum` raises `ValueError` instead.

This is the last lesson where one neuron is one object. From lesson 4 on, the
unit is a *layer* of neurons stored as one matrix, because a thousand neuron
objects would mean a thousand slow Python loops.

**Why pass in a random generator?** `np.random.default_rng(0)` produces the same
"random" numbers every run. A bug that appears one run in five is very hard to
chase; a reproducible one is easy. Functions in this course never create their
own randomness: it is always passed in.
