# Lesson 0 — Theory: derivatives, partial derivatives, the chain rule

Training a neural network is one question asked millions of times:

> If I turn this knob a little, does the error go up or down, and by how much?

The answer is a derivative. This lesson builds three tools for computing it:
the plain derivative (one input), the partial derivative (many inputs), and the
chain rule (many steps). Section 7 combines all three into backpropagation on
the smallest possible network.

---

## 1. A derivative is a sensitivity

**Analogy: the shower tap.** Turn the tap a tiny amount and the water
temperature changes. The derivative is the ratio, degrees of temperature per
degree of turn, *at the tap's current position*.

- Large: a touchy tap. A small turn makes a big change.
- Zero: turning does nothing right now.
- Negative: turning it up makes the water colder.

The words "at the current position" matter. The same tap can be touchy near
the middle and dead near the end of its range. A derivative is a property of a
function **at a point**, not of the function as a whole.

**Math.** For a function `f` of one number:

```
df/dx  =  how much f changes per unit change in x, at this x
       ≈  (change in f) / (small change in x)
```

## 2. Computing it numerically: the central difference

The definition suggests a method: nudge `x` by a tiny `h` and measure.

```
one-sided:   df/dx ≈ ( f(x + h) − f(x) ) / h
central:     df/dx ≈ ( f(x + h) − f(x − h) ) / 2h      ← use this one
```

The central version looks both ways, and its error shrinks much faster as `h`
shrinks. Worked example, `f(x) = x²` at `x = 1` with a deliberately large `h = 0.5`:

| Method | Calculation | Estimate | True value |
|---|---|---|---|
| One-sided | (1.5² − 1²) / 0.5 | 2.5 | 2 |
| Central | (1.5² − 0.5²) / 1.0 | **2.0** | 2 |

**In code.** Python functions are values, so you can hand one to another function:

```python
from collections.abc import Callable

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    ...

numerical_derivative(lambda x: x * x, 3.0)    # ≈ 6.0
```

The numerical derivative is slow (two calls to `f` per input) and approximate,
but it needs no maths and works on any function. That makes it the perfect
**referee**: every gradient you write by hand in this course is tested against it.

## 3. Three ways to get a derivative

| Way | How | Strength | Weakness | Where |
|---|---|---|---|---|
| Numerical | Nudge and measure | Works on anything | Slow, approximate | Lesson 0; every test file |
| Analytic | Derive the formula by hand, type it in | Exact, fast | Redo it for every new function | Lessons 0–7 |
| Automatic | The program records its own arithmetic and applies the chain rule for you | Exact, fast, no hand work | Machinery to build | Lesson 8, then PyTorch |

You only need two analytic results in this lesson: the derivative of `x²` is
`2x`, and the derivative of the sigmoid (section 5).

## 4. Partial derivatives: several inputs, nudge one

**Concept.** A function of one input has one slope. A function of several inputs
has one slope **per input**. To get the slope for one input, freeze all the
others and treat them as constants.

**Analogy: a hillside.** Your altitude depends on how far east and how far north
you stand. There are two slopes: the one you feel stepping east, and the one you
feel stepping north. Each is measured while the other direction stays put.

**Notation.** The curly ∂ means "the other inputs are held still":

```
∂f/∂x  =  how fast f changes as x moves, with every other input frozen
```

**Worked example.** `f(x, y) = x²y + 3y` at the point `(2, 5)`, where `f = 35`.

- **∂f/∂x:** treat `y` as a fixed number. `x²y` becomes `2xy`, and `3y` contains
  no `x`, so it contributes 0. Result: `2xy = 20`.
- **∂f/∂y:** treat `x` as a fixed number. `x²y` becomes `x²`, and `3y` becomes `3`.
  Result: `x² + 3 = 7`.

Check by nudging each input on its own:

| Nudge | f after | Change in f | Predicted: slope × nudge |
|---|---|---|---|
| x: 2 → 2.01 (y stays 5) | 35.2005 | 0.2005 | 20 × 0.01 = 0.20 |
| y: 5 → 5.01 (x stays 2) | 35.0700 | 0.0700 | 7 × 0.01 = 0.07 |

**The gradient** is the list of every partial derivative, in input order:

```
∇f(2, 5) = [∂f/∂x, ∂f/∂y] = [20, 7]
```

It has one entry per input: the same length as the inputs. (From lesson 3 on,
"the same length" becomes "the same shape", and it is the most useful fact in the
course.) In a neural network the inputs of interest are the **weights**, the
function is the **loss**, and the gradient says how sensitive the loss is to
every weight at once.

**Three ways.**

| Math | Shape | Code |
|---|---|---|
| ∂f/∂xᵢ | one number | `partial_derivative(f, args, i)` |
| ∇f = [∂f/∂x₀, …, ∂f/∂xₙ₋₁] | n numbers, like `args` | `numerical_gradient(f, args)` |

## 5. The chain rule: several steps, multiply the slopes

**Concept.** Networks are functions inside functions. When `x` feeds `u` and `u`
feeds `y`, a nudge to `x` gets scaled once at each step, so the overall
sensitivity is the **product** of the step sensitivities.

**Analogy: gears.** Gear A drives B, B drives C. B turns 2 times per turn of A;
C turns 3 times per turn of B. So C turns 2 × 3 = 6 times per turn of A.

```
   x ──[ slope 2 ]──▶ u ──[ slope 3 ]──▶ y          dy/dx = 2 × 3 = 6
```

**Why it is a product.** Up close, any smooth function looks like a straight
line, so a small nudge is just scaled by the slope:

```
u = g(x)      Δu ≈ g′(x) · Δx
y = f(u)      Δy ≈ f′(u) · Δu  =  f′(u) · g′(x) · Δx

divide by Δx:           dy/dx = f′(u) · g′(x)  =  dy/du · du/dx
```

**The one trap.** `f′(u)` is measured at `u = g(x)`, the value the inner step
actually produced, not at `x`.

**Worked example.** `y = (3x + 1)²` at `x = 2`.

| Step | Value | Local slope |
|---|---|---|
| inner: u = 3x + 1 | u = 7 | du/dx = 3 |
| outer: y = u² | y = 49 | dy/du = 2u = **14** (at u = 7, not at x = 2) |
| total | | dy/dx = 14 × 3 = **42** |

**The sigmoid's derivative, as a chain-rule exercise.**

```
σ(z) = 1 / (1 + e⁻ᶻ)  =  (1 + e⁻ᶻ)⁻¹

outer: u⁻¹, slope −u⁻²          inner: u = 1 + e⁻ᶻ, slope −e⁻ᶻ

dσ/dz = −(1 + e⁻ᶻ)⁻² · (−e⁻ᶻ)  =  e⁻ᶻ / (1 + e⁻ᶻ)²  =  σ(z) · (1 − σ(z))
```

The last step is algebra worth doing once on paper. The result is why
neural-network code stores the sigmoid's *output*: if `a = σ(z)`, the slope is
simply `a(1 − a)`, with no second exponential. At `z = 0`, `a = 0.5` and the
slope is 0.25, its maximum. Far from zero the slope is nearly 0. Remember that:
it comes back as a real problem in lesson 6.

**Three ways.**

| Math | Shape | Code |
|---|---|---|
| d/dx f(g(x)) = f′(g(x)) · g′(x) | one number | `chain_rule(d_outer, inner, d_inner, x)` |

## 6. Several paths: multiply along, add across

**Concept.** Sometimes an input reaches the output by more than one route. Each
route contributes its own chain-rule product, and the contributions **add up**.

```
              ┌──▶ x = t² ──┐
         t ───┤              ├──▶ z = x · y
              └──▶ y = 3t ──┘
```

```
dz/dt  =  ∂z/∂x · dx/dt   +   ∂z/∂y · dy/dt
          └─ path via x ─┘    └─ path via y ─┘
```

**Worked example**, at `t = 2` (so `x = 4`, `y = 6`):

| Path | Local slopes | Product |
|---|---|---|
| via x | ∂z/∂x = y = 6, dx/dt = 2t = 4 | 24 |
| via y | ∂z/∂y = x = 4, dy/dt = 3 | 12 |
| total | | **36** |

Check by substituting first: `z = t² · 3t = 3t³`, so `dz/dt = 9t² = 36`. ✓

**Why this matters.** This one rule — *multiply along each path, add across
paths* — is all of backpropagation. Every weight's gradient is a sum of products
of local slopes along every path from that weight to the loss. In lesson 5 a
hidden neuron feeds the output; in lesson 6 an input feeds every neuron; in
lesson 8 one number is reused in many places. The adding is the same each time.

**Three ways.**

| Math | Shape | Code |
|---|---|---|
| Σ over paths of Π (slopes along the path) | one number | `sum_over_paths([[6, 4], [4, 3]])` |

## 7. Putting it together: backpropagation on one weight

The smallest neural network: one input `x`, one weight `w`, a sigmoid, and a
loss comparing the output with the target `y`.

```
            forward ─────────────────────────────────────────────▶

   w ──▶ [ z = w·x ] ──▶ [ a = σ(z) ] ──▶ [ L = (a − y)² ]
   x ───────┘                                    y ───┘

            ◀───────────────────────────────────────────── backward
              dz/dw = x       da/dz = a(1 − a)     dL/da = 2(a − y)
```

The **forward pass** computes and records `z`, `a` and `L`. The **backward pass**
multiplies the local slopes, starting from the loss and working back to `w`,
using the values the forward pass recorded.

**Worked example.** `w = 0.5`, `x = 2`, `y = 1`.

| Forward | Value |
|---|---|
| z = w·x | 1 |
| a = σ(z) | 0.7311 |
| L = (a − y)² | 0.0723 |

| Backward link | Local slope | Value |
|---|---|---|
| L from a | 2(a − y) | −0.5379 |
| a from z | a(1 − a) | 0.1966 |
| z from w | x | 2 |
| **dL/dw** | product | **−0.2115** |

**Reading the answer.** The slope is negative: increasing `w` decreases the
loss. So the next step is to *raise* `w` a little — subtract a small multiple of
the slope:

```
w_new = w − η · dL/dw  =  0.5 − 0.5 × (−0.2115)  =  0.6058
```

The loss at the new `w` is lower. That single line, repeated, is gradient
descent: lesson 2. Everything after that is about doing it for millions of
weights at once.

**Why record the forward values?** The backward pass needs `a` (and `x`). Saving
them means the backward pass never recomputes anything. This split — forward
saves, backward reads — is the structure of every layer you write from lesson 6
on, and of PyTorch itself.

**Three ways.**

| Math | Shape | Code |
|---|---|---|
| dL/dw = 2(a − y) · a(1 − a) · x | one number | `tiny_backward(tiny_forward(w, x, y), x, y)` |

---

## Summary

| Tool | Question it answers | Rule |
|---|---|---|
| Derivative | How sensitive is the output to the input? | Nudge and measure, or use a known formula |
| Partial derivative | …to *one* of several inputs? | Freeze the others |
| Gradient | …to *each* input? | One partial per input, same length as the inputs |
| Chain rule | …through a chain of steps? | Multiply the local slopes |
| Several paths | …when there are several routes? | Multiply along each, add across |
