# Lesson 0 — Derivatives, partial derivatives, the chain rule

**Read [THEORY.md](THEORY.md) first.** Each task below points at the section it uses.

## What you build

Six small functions in `derivatives.py` that compute derivatives three ways, and
finish by running backpropagation on a network with a single weight. No NumPy
yet: everything is a plain `float`.

## Setup

```bash
pytest lesson00_derivatives -x      # run from the repository root
```

Every test fails at first with `NotImplementedError: TODO`. Work through the
tasks in order; the tests are in the same order.

## Tasks

| # | Function(s) | Theory | What it does |
|---|---|---|---|
| 1 | `numerical_derivative` | §2 | The central difference: the referee for the whole course |
| 2 | `d_square`, `d_sigmoid` | §3, §5 | Two exact derivatives, checked against task 1 |
| 3 | `chain_rule` | §5 | The chain rule as a function of functions |
| 4 | `partial_derivative`, `numerical_gradient` | §4 | One input at a time; then all of them |
| 5 | `sum_over_paths` | §6 | Multiply along each path, add across paths |
| 6 | `tiny_forward`, `tiny_backward` | §7 | Forward pass records values; backward pass multiplies slopes |

**Types to know.** `ScalarFn` is any function from one `float` to a `float`.
`MultiFn` takes a sequence of floats (a list or tuple) and returns a `float`.
`Trace` is a small frozen dataclass holding `z`, `a` and `L`. Hover over any
name in your editor to see its type.

## Experiments

Once the tests pass, open a Python prompt in this folder (`cd lesson00_derivatives; python`):

1. **The limits of nudging.** The true derivative of `x³` at 2 is 12.

   ```python
   from derivatives import *
   for h in (1e-1, 1e-3, 1e-5, 1e-8, 1e-11, 1e-14):
       print(h, numerical_derivative(lambda x: x**3, 2.0, h))
   ```
   Why does a *smaller* `h` eventually give a *worse* answer? (Floats carry about
   16 significant digits. Subtracting two nearly equal numbers throws most away.)

2. **Gradient descent by hand.** Start at `w = 0.5, x = 2, y = 1` and repeat
   `w = w - 0.5 * tiny_backward(tiny_forward(w, x, y), x, y)` ten times, printing
   `w` and the loss. What happens to the size of each step as the loss falls? Why?

3. **A vanishing slope.** Try the same loop from `w = -5`. The loss is large but
   `w` barely moves. Look at `trace.a` and work out which of the three local slopes
   is responsible (THEORY §5, last paragraph).

## The limit

You can now compute how any knob affects any output. But there is nothing with
real knobs yet: one weight, one input, no bias, no notion of a dataset. Lesson 1
builds an actual neuron.

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1:** two calls to `f`, one subtraction, one division.
- **Task 3:** `d_outer` is called on `inner(x)`, not on `x`. That is the whole trap.
- **Task 4:** `up = list(args)` makes a copy you can change without touching the
  caller's list. Make two copies, nudge one up and one down.
- **Task 5:** a running total over paths, and inside it a running product over
  slopes. An empty path list should give 0.
- **Task 6:** in `tiny_backward`, compute the three slopes as named variables
  (`dL_da`, `da_dz`, `dz_dw`) and multiply them. Read `trace.a`; do not call `sigmoid`.
</details>
