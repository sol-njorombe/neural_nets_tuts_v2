# Lesson 5 — Theory: hidden layers and backpropagation

This is the lesson the first half of the course has been building towards: the
three-input XOR network, starting from random weights, trained with
backpropagation. With lessons 3 and 4 behind you it is two more lines going
forward and four more going back.

## 1. A hidden layer is a team of specialists

One neuron makes one flat cut. Give the problem to a small team instead:

```
 inputs              hidden layer            output
                    ┌─▶ h₀ ──┐
 x₀ ──┐             │        │
 x₁ ──┼── every ────┼─▶ h₁ ──┼──── combines ──▶ a
 x₂ ──┘   input to  │        │     their
          every h   └─▶ h₂ ──┘     opinions
```

Each **hidden** neuron looks at the raw inputs and makes its own cut. The
**output** neuron never sees the inputs; it sees only the hidden neurons'
outputs and combines them. Several flat cuts combined can carve out shapes no
single cut can.

They are called hidden because no dataset says what they should compute. There
is no column for "what h₁ should output". Training has to invent useful roles
for them.

## 2. XOR by hand

Before training anything, build it yourself (task 1). Three specialists:

| Hidden neuron | Fires when | Weights | Bias |
|---|---|---|---|
| h₀ | at least one input is on | 20, 20, 20 | −10 |
| h₁ | at least two are on | 20, 20, 20 | −30 |
| h₂ | all three are on | 20, 20, 20 | −50 |

With weights of 20, each sum is 0, 20, 40 or 60, so each bias puts the hurdle
halfway between two of those, and the sigmoid outputs are very close to 0 or 1.

Now count the inputs that are on and see what `h₀ − h₁ + h₂` gives:

| Inputs on | h₀ | h₁ | h₂ | h₀ − h₁ + h₂ | XOR3 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 1 | 1 |
| 2 | 1 | 1 | 0 | 0 | 0 |
| 3 | 1 | 1 | 1 | 1 | 1 |

So the output neuron needs weights `[20, −20, 20]` and a bias of −10. The
network computes XOR3 exactly, and every piece is a single neuron of the kind
lesson 1 built. The hidden layer turned an impossible problem into an easy one
by **re-describing the inputs** in terms the output neuron can cut cleanly. That
is the general idea behind every deep network.

## 3. The forward pass

Each layer is lesson 4's batched neuron with more output columns:

```
X  (n, d_in)  @ W1 (d_in, d_hidden) + b1 (d_hidden,)  →  Z1 (n, d_hidden)    H = σ(Z1)
H  (n, d_hidden) @ W2 (d_hidden, 1) + b2 (1,)         →  Z2 (n, 1)           A = σ(Z2)
```

Column j of `W1` holds hidden neuron j's weights, so `H[k, j]` is "hidden neuron
j's opinion of example k".

| Math | Shape | Code |
|---|---|---|
| H = σ(X W₁ + b₁) | (n, d_hidden) | `H = sigmoid(X @ W1 + b1)` |
| A = σ(H W₂ + b₂) | (n, 1) | `A = sigmoid(H @ W2 + b2)` |

## 4. Backpropagation: passing blame backwards

**Analogy.** A product ships with a defect. The manager (the output neuron) is
clearly responsible and takes blame in proportion to the error. The manager
then turns to the team: "I relied on each of you by a certain amount, my weight
on you. Your share of my blame is scaled by that." Each team member then
adjusts their own inputs' weights in proportion to their share.

In matrices, the output layer is exactly lesson 4 with `H` as its input:

```
dA  = 2(A − Y) / n             (n, 1)
dZ2 = dA · A(1 − A)            (n, 1)           δ for the output neuron
dW2 = H.T @ dZ2                (d_hidden, 1)
db2 = dZ2.sum(axis=0)          (1,)
```

The new step is passing blame to the hidden layer. `Z2 = H @ W2 + b2` is a
matrix multiply, and lesson 3 §8 says how blame passes through one to its input:

```
dH  = dZ2 @ W2.T               (n, 1) @ (1, d_hidden) → (n, d_hidden)
```

Row k, column j: example k's output blame times the weight from hidden neuron j.
That is the manager's blame scaled by reliance. Then the hidden layer is the
same three lines again, with `X` as its input:

```
dZ1 = dH · H(1 − H)            (n, d_hidden)    δ for every hidden neuron
dW1 = X.T @ dZ1                (d_in, d_hidden)
db1 = dZ1.sum(axis=0)          (d_hidden,)
```

**The pattern** is the whole algorithm, at any depth:

```
for each layer, from the last back to the first:
    δ   = (blame on my output) · (my activation's slope)
    dW  = (my input).T @ δ
    db  = δ summed over rows
    blame on my input = δ @ W.T           ← hand this to the layer before
```

| Math | Shape | Code |
|---|---|---|
| ∂L/∂H = δ₂ W₂ᵀ | (n, d_hidden) | `dH = dZ2 @ W2.T` |
| δ₁ = ∂L/∂H ⊙ H(1 − H) | (n, d_hidden) | `dZ1 = dH * H * (1 - H)` |
| ∂L/∂W₁ = Xᵀ δ₁ | (d_in, d_hidden), like W1 | `dW1 = X.T @ dZ1` |

(⊙ means elementwise multiplication.)

## 5. Worked example: one input, one hidden neuron, one example

Small enough to do with a calculator: `x = 1`, `y = 1`, `W1 = [[1]]`,
`W2 = [[2]]`, both biases 0.

| Forward | Value |
|---|---|
| h = σ(1 · 1) | 0.7311 |
| a = σ(2 · 0.7311) | 0.8119 |
| L = (a − y)² | 0.0354 |

| Backward | Calculation | Value |
|---|---|---|
| dA | 2(0.8119 − 1) / 1 | −0.3763 |
| dZ2 | −0.3763 × 0.8119 × 0.1881 | −0.0575 |
| **dW2** | h × dZ2 = 0.7311 × −0.0575 | **−0.0420** |
| dH | dZ2 × W2 = −0.0575 × 2 | −0.1150 |
| dZ1 | −0.1150 × 0.7311 × 0.2689 | −0.0226 |
| **dW1** | x × dZ1 | **−0.0226** |

Both slopes are negative: increasing either weight would bring the output closer
to 1. Note how the blame shrinks as it travels back: each sigmoid slope is at
most 0.25. Lesson 6 returns to that.

## 6. Training, and what can go wrong

Training is lesson 4's loop with four parameters instead of two. With 4 hidden
neurons and `lr = 2`, the network learns XOR3 from every seed the tests try.

Two things that are not bugs:

- **Getting stuck.** With only 2 hidden neurons, about half of random starts end
  at 50–88% accuracy and stay there. The loss surface of a network has many
  valleys, and where you start decides which one you roll into. More hidden
  neurons usually means more routes to a good solution (experiment 2).
- **Symmetry.** If every weight starts at the same value, every hidden neuron
  computes the same thing, receives the same blame, and stays identical forever:
  the team has one opinion, copied. Random initial weights break the symmetry.

## 7. The limit

Look at `backward`: it is written for exactly two layers, both sigmoid, one
output, mean squared error. Three layers? A different activation? Ten outputs
for ten digit classes? Each means rewriting `forward` and `backward` together,
and one slip gives a network that quietly learns badly instead of crashing.

But §4 showed every layer does the same thing: forward saves its input, backward
takes blame on its output and returns blame on its input. Lesson 6 turns that
into a reusable `Layer`, and the network becomes a list.
