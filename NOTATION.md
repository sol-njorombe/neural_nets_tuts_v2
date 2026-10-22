# Notation

One meaning per symbol, for the whole course. Code uses the same names as the math.

## Scalars (lessons 0–2)

| Symbol | Code | Meaning |
|---|---|---|
| x, xᵢ | `x`, `x[i]` | An input, or the i-th input |
| y | `y` | The target: the answer we want |
| w, wᵢ | `w`, `w[i]` | A weight, or the weight on input i |
| b | `b` | The bias |
| z | `z` | The weighted sum, before the activation: z = Σ wᵢxᵢ + b |
| σ | `sigmoid` | The sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ) |
| a | `a` | The activation: the neuron's output, a = σ(z) |
| L | `L` | The loss: one number saying how wrong we are |
| η | `lr` | The learning rate: how big a step to take |
| h | `h` | A tiny nudge, used to estimate derivatives numerically |

## Arrays (lesson 3 onward)

Rows are always examples.

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| n | `n` | — | Number of examples (rows) in the batch |
| d_in, d_out | `d_in`, `d_out` | — | Number of inputs to, and outputs from, a layer |
| X | `X` | (n, d_in) | Inputs, one example per row |
| Y | `Y` | (n, d_out) | Targets, one example per row. A single target is still a column: (n, 1) |
| W | `W` | (d_in, d_out) | Weights: column j holds neuron j's weights |
| b | `b` | (d_out,) | Biases, one per neuron |
| Z | `Z` | (n, d_out) | Weighted sums: Z = X @ W + b |
| A | `A` | (n, d_out) | Activations: A = σ(Z) |

## Derivatives

| Symbol | Code | Meaning |
|---|---|---|
| df/dx | `d_f` | The derivative of a one-input function f |
| ∂f/∂x | `partial_derivative(f, args, i)` | The derivative of f with respect to one input, others held still |
| ∇f | `grad` | The gradient: the list of all partial derivatives of f |
| ∂L/∂W | `dW` | The gradient of the loss with respect to W. **Always the same shape as W.** |
| δ | `delta` | ∂L/∂z: how the loss changes with a neuron's weighted sum. The "blame" passed backwards. |

## Classification (lesson 6 onward)

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| k | `k` | — | Number of classes |
| logits | `logits` | (n, k) | Raw class scores from the last layer, any real number |
| P | `P` | (n, k) | Probabilities from softmax; each row sums to 1 |
| y (labels) | `labels`, `y` | (n,) | The correct class for each example, an integer in 0..k−1 |
| one_hot(y) | `one_hot(y, k)` | (n, k) | 1 at each row's correct class, 0 elsewhere |

## Optimisers (lesson 7 onward)

| Symbol | Code | Meaning |
|---|---|---|
| β | `beta` | Momentum's decay: how much of the old velocity to keep |
| v (momentum) | `velocities` | Running sum of past gradients |
| m, v (Adam) | `m`, `v` | Running means of the gradient and of its square |
| β₁, β₂ | `beta1`, `beta2` | Adam's decay rates for m and v |
| t | `t` | Number of optimiser steps taken |
| ε | `eps` | A tiny number that prevents division by zero |
