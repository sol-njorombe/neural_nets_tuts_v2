# Lesson 5 — Hidden layers and backpropagation

**Read [THEORY.md](THEORY.md) first.**

## What you build

A network with a hidden layer that learns three-input XOR from random weights,
the problem lessons 2 and 4 proved one neuron cannot solve. First you build the
XOR network by hand, so you know a solution exists and what one looks like.

## Setup

```bash
pytest lesson05_backprop -x
python lesson05_backprop/backprop.py      # after task 4: training, then the hidden layer's view of each input
```

## The types

| Type | Fields | Shapes |
|---|---|---|
| `TwoLayerParams` | `W1`, `b1`, `W2`, `b2` | `(d_in, d_hidden)`, `(d_hidden,)`, `(d_hidden, 1)`, `(1,)` |
| `Cache` | `X`, `H`, `A` | `(n, d_in)`, `(n, d_hidden)`, `(n, 1)` |
| `Grads` | `dW1`, `db1`, `dW2`, `db2` | the same shapes as the parameters |

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `HAND_XOR`: choose every weight yourself | §2 |
| 2 | `forward`: two layers | §3 |
| 3 | `backward`: output layer, pass blame back, hidden layer | §4, §5 |
| 4 | `train` | §6 |

## Experiments

1. **What did the hidden layer learn?** The demo prints `H` for every input.
   Can you describe any hidden neuron's job in words, like the ones in §2? (Often
   they are less tidy than the hand-built ones, but they do the job.)
2. **Team size.** Train with `d_hidden` = 1, 2, 3 and 8 over seeds 0–9 and count
   how many reach 100%. Why does 1 never work? Why is 2 unreliable?
3. **Symmetry.** Set every entry of W1 and W2 to 0.5 and both biases to 0, then
   train. Print `H` afterwards. What do you notice about its columns?
4. **Blame fades.** Print `np.abs(g.dW2).mean()` and `np.abs(g.dW1).mean()` at
   the first epoch. Which layer gets the smaller gradients, and why (§5)?

## The limit

`backward` only works for this exact architecture. Lesson 6 makes layers into
bricks with a uniform `forward`/`backward` interface, so that a network is a
list and any stack of bricks gets correct gradients automatically. It also moves
from XOR to reading handwritten digits.

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1:** follow the tables in THEORY §2. `np.full((3, 3), 20.0)` makes W1.
  W2 is a column: `np.array([[20.0], [-20.0], [20.0]])`.
- **backward:** unpack `X, H, A = cache.X, cache.H, cache.A`. Write the output
  layer first and run only the W2/b2 gradient tests (`-k W2`).
- If dW2 passes and dW1 fails, check `dH = dZ2 @ params.W2.T` and that the hidden
  sigmoid slope uses `H`, not `A`.
- **train:** update all four arrays with `-=`.
</details>
