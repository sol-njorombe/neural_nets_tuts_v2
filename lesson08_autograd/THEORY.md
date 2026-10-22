# Lesson 8 — Theory: automatic differentiation

Short, and probably the most satisfying lesson. You write about 60 lines and
never derive a gradient by hand again. It is also exactly the idea at the heart
of PyTorch, so lesson 9 will feel familiar.

## 1. The problem

Every `backward` you have written was the chain rule applied to a handful of
additions, multiplications and activation functions. You had to derive it by
hand for each new kind of layer, and keep it in sync with `forward`. One slip
gives a network that learns badly without crashing; only the gradient checks
caught those.

Lesson 0 §3 listed three ways to get derivatives: numerical (slow), analytic
(hand work), and **automatic**. This lesson builds the third.

## 2. Numbers that keep receipts

Until now, `c = a * b` gave you `c` and forgot where it came from. Instead, make
each number keep a receipt. A `Value` holds:

| Field | Meaning |
|---|---|
| `data` | the number itself |
| `children` | the Values it was computed from |
| `_backward` | a small function: how to pass my blame to my children |
| `grad` | ∂(final result)/∂(this Value), filled in by `backward()` |

Doing arithmetic **is** the forward pass, and it builds a graph as a side effect:

```python
x, w, b = Value(2.0), Value(3.0), Value(-4.0)
L = (x * w + b) ** 2      # L.data == 4.0, and L remembers the whole recipe
L.backward()               # walk the receipts in reverse
w.grad                     # 8.0, with no formula written by you
```

```
   x ─┐
      ├─[*]── u ─┐
   w ─┘          ├─[+]── s ──[**2]── L
   b ────────────┘
```

## 3. Each operation knows only its own local slope

That is the key. No operation knows about the whole network; each one knows how
its output changes with its inputs, which you worked out in lesson 0:

| Operation | Local slope(s) |
|---|---|
| out = a + b | 1 for a, 1 for b |
| out = a × b | b for a, a for b |
| out = aⁿ (n a plain number) | n·aⁿ⁻¹ |
| out = eᵃ | eᵃ, which is out itself |
| out = tanh(a) | 1 − out² |
| out = max(0, a) | 1 if a > 0, else 0 |

In `_backward`, each child receives `local slope × out.grad`: the chain rule,
one link at a time. That is the same move as `δ × input` since lesson 2.

Everything else is built from these. Subtraction is `a + (−1)·b`. Division is
`a × b⁻¹`. The sigmoid is `1 / (1 + e⁻ˣ)`, and its derivative `a(1 − a)` falls
out automatically, with no one writing it down.

## 4. Running it backwards

`backward()` on the final result:

1. Sets its own `grad` to 1 (∂L/∂L = 1).
2. Visits every Value in the graph from the end back to the start, calling each
   one's `_backward`, which adds to its children's `grad`.

**Worked example**, `L = (x·w + b)²` with `x = 2`, `w = 3`, `b = −4`:

| Forward | Value |
|---|---|
| u = x·w | 6 |
| s = u + b | 2 |
| L = s² | 4 |

| Backward (in reverse) | Calculation | Result |
|---|---|---|
| L.grad | start | 1 |
| s.grad | 2s × L.grad = 4 × 1 | 4 |
| u.grad, b.grad | 1 × s.grad | 4, 4 |
| w.grad | x × u.grad = 2 × 4 | **8** |
| x.grad | w × u.grad = 3 × 4 | 12 |

### Two traps the tests check

**Accumulate, never overwrite.** In `a + a`, the Value `a` is used twice and
must collect blame from both uses: its gradient is 2. Always `+=` on `.grad`.
This is lesson 0 §6 again: paths add.

**Order matters.** A Value must receive all of its blame before passing any on.
In `b = 2a; c = b × b`, `b` gets blame from both sides of the multiplication.
If `b`'s `_backward` ran after hearing from only one, `a` would get half its
gradient. So `backward()` first sorts the graph so that every Value comes
**after** the Values it was computed from (a **topological sort**), then runs
`_backward` in reverse of that order:

```
visit(v):
    if v already visited: return
    mark v visited
    for each child of v: visit(child)
    append v to the order          ← only after all its children
```

## 5. A whole network with no backward methods

`tiny_nn.py` (given) builds `Neuron`, `Layer` and `MLP` classes on your `Value`.
Read it: there is no `backward` method anywhere. Training is:

```python
loss = mean of (model(x) − y)² over the data    # builds the graph
for p in parameters: p.grad = 0                  # clear last step's blame
loss.backward()                                  # one call
for p in parameters: p.data −= lr * p.grad
```

The payoff test builds lesson 5's exact network on all of XOR3 from Values and
checks that `w.grad` equals what your lesson 5 formulas computed. Then it trains
a network with **two** hidden layers. In lesson 5, adding that second layer
meant new calculus. Here it is `[4, 4, 1]` instead of `[4, 1]`.

**Why clear `.grad` every step?** Because `backward` accumulates (`+=`). Without
clearing, each step's gradient would be added to every previous step's. PyTorch
has exactly the same rule, which is why every PyTorch training loop calls
`optimizer.zero_grad()`.

## 6. The limit

Time `tiny_nn.py`. XOR3 with 41 parameters takes about two seconds. Every single
multiplication creates a Python object and a closure; for the digit network,
with about 2,400 parameters and 1,437 images, one epoch would build millions of
them.

What you want is this lesson's idea applied to lesson 5's arrays: a Value that
wraps a **whole matrix**, where `X @ W` is one node with one local rule
(lesson 3 §8's table), running in optimised C++ or on a GPU. That is, almost
exactly, a PyTorch tensor. Lesson 9.
