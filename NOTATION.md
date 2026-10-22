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
