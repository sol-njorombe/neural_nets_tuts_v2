# Lesson 8 — Autograd

**Read [THEORY.md](THEORY.md) first.**

## What you build

A `Value` class: a number that records the arithmetic done with it, and can then
compute the gradient of any expression built from it. Then a given network
library, written on top of your `Value` with no backward methods at all, learns
XOR with two hidden layers.

## Setup

```bash
pytest lesson08_autograd -x
python lesson08_autograd/tiny_nn.py      # after task 3
```

## Tasks

`__add__` is a fully worked example. Read it line by line first.

| # | What | Theory |
|---|---|---|
| 1 | `__mul__` | §3 |
| 2 | `__pow__`, `exp`, `tanh`, `relu` | §3 |
| 3 | `topological_order`, `backward` | §4 |

Subtraction, division, negation and `sigmoid` are given, built from your
operations. Once task 1 passes, `-v` and `a - b` start working; once task 2
passes, `a / b` and `v.sigmoid()` do too.

## Experiments

1. **Print the graph.** Write a short function that walks `topological_order()`
   and prints each Value's `op`, `data` and `grad`. Use it on the worked example
   in THEORY §4 and check every row of the table.
2. **Compare with lesson 0.** Build `tiny_forward` from lesson 0 out of Values
   (`w * x`, `.sigmoid()`, `** 2`) and check `w.grad` against `tiny_backward`.
3. **Forget to clear.** Remove `p.grad = 0.0` from `tiny_nn.train`. What happens
   to training, and why?
4. **Add an operation.** Add `log` (slope `1/a`) and write cross-entropy for one
   example with two classes. Check the gradient against lesson 6's `P − one_hot`.
5. **Speed.** Time one epoch of `tiny_nn.train` on XOR3, then estimate how long
   one epoch on the digits would take.

## The limit

Correct, general, and slow: one Python object per number. Lesson 9 moves to
PyTorch, whose tensors are this lesson's idea applied to whole arrays, in
optimised native code. You will recognise `loss.backward()` and `.grad`.

<details>
<summary>Hints (open only after trying)</summary>

- **Every operation** follows `__add__`'s shape: build `out`, define `_backward`
  inside the method, attach it with `out._backward = _backward`, return `out`.
- **One-input operations** pass `(self,)` as the children: note the comma.
- **exp / tanh:** the slope uses `out.data`, which is already computed.
- **topological_order:** a nested `visit` function and a `visited` set; append
  after the loop over children.
- **backward:** don't forget `self.grad = 1.0` first, or every gradient stays 0.
</details>
