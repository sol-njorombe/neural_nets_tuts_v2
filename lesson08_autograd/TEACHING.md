# Lesson 8 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Explain reverse-mode automatic differentiation as "each operation knows its local slope; backward multiplies along paths and adds across them".
2. Implement a Value with arithmetic, activation functions and backward.
3. Explain why gradients accumulate with `+=`, and why they must be cleared each step.
4. Explain why the graph must be visited in reverse topological order.
5. Explain why scalar autograd is slow, and what tensor autograd changes.

## Suggested session (about 1.5 hours)

| Time | Activity |
|---|---|
| 10 min | Receipts analogy; draw THEORY §2's graph |
| 10 min | Learner fills in THEORY §4's backward table by hand |
| 45 min | Tasks 1–3 |
| 20 min | Read `tiny_nn.py` together; experiments 1 and 3 |
| 5 min | Discussion |

## Discussion questions

1. Where is the chain rule in this code? (In every `_backward`: local slope × `out.grad`.)
2. Where is "add over paths"? (`+=` on `.grad`.)
3. Why does `backward()` need a sort, when `forward` did not? (Forward runs in the order you wrote the code; backward must reverse it, and must wait for every use of a Value.)
4. `tiny_nn.py` has no backward methods. Who derived the gradient of tanh? Of the whole network?
5. What would make this fast? (One node per matrix operation instead of per number; native code.)

## Common mistakes

| Symptom | Likely cause |
|---|---|
| `a + a` gives grad 1 | `=` instead of `+=` in `_backward` |
| Every gradient 0 | Forgot `self.grad = 1.0` in `backward` |
| `b * b` example gives 12 instead of 24 | Visited in the wrong order, or a Value appears twice in the order |
| `exp` slope wrong | Used `self.data` instead of `out.data` (or `math.exp(self.data)`) |
| `TypeError: 'Value' object is not iterable` | Wrote `(self)` instead of `(self,)` for children |

## Checks for understanding

- "f = a·b + a with a = 2, b = 5. Gradient of a?" (b + 1 = 6.)
- "Why does PyTorch need `optimizer.zero_grad()`?"
- "Which local slope does `relu` use at exactly 0?" (This code uses 0; either choice works in practice.)
