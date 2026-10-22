# Lesson 0 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Explain a derivative as a sensitivity at a point, without formulas.
2. Estimate any derivative numerically, and explain why the central difference beats the one-sided one.
3. Compute a partial derivative by freezing the other inputs, and say what the gradient is.
4. Apply the chain rule, including evaluating the outer derivative at the inner value.
5. Combine several paths by adding their products.
6. Run a forward and backward pass on a one-weight network and say which way to move the weight.

## Suggested session (about 2 hours)

| Time | Activity |
|---|---|
| 15 min | Shower-tap and gears analogies on a whiteboard. No formulas yet. |
| 15 min | Learner does THEORY §4 and §6 worked examples by hand, on paper |
| 60 min | Tasks 1–6 |
| 20 min | Experiments 1–3 |
| 10 min | Discussion questions |

## Discussion questions

1. A derivative is zero at some point. Does that mean the function is flat everywhere? What does it mean for a weight in a network?
2. Why do we trust the numerical derivative as a referee, when it is only approximate?
3. In §6, what would go wrong if we *multiplied* the two path contributions instead of adding them? (Try it with the numbers.)
4. In experiment 3, the loss is large but the weight barely moves. Is that a bug in the code or a property of the network?
5. `tiny_backward` takes a `Trace` instead of recomputing the forward pass. What would it cost to recompute, for a network with a million weights?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| `chain_rule` test gives 18 instead of 42 | Called `d_outer(x)` instead of `d_outer(inner(x))` |
| `partial_derivative` passes alone but the "unchanged inputs" test fails | Nudged `args` in place instead of a copy |
| `d_sigmoid` off by a sign | Wrote `a * (a - 1)` |
| `tiny_backward` sign flipped | Wrote `2 * (y - a)` instead of `2 * (a - y)` |
| Central-difference test fails with 2.5 | One-sided difference |

## Checks for understanding

Ask the learner, without code:

- "At `x = 3`, the derivative of `f` is −2. If I increase `x` by 0.01, roughly how does `f` change?" (Falls by about 0.02.)
- "`f(x, y) = 5x + y²`. What is the gradient at `(1, 3)`?" (`[5, 6]`)
- "Three gears with ratios 2, 0.5 and 4. Overall ratio?" (4)
- "If `dL/dw` is positive, should `w` go up or down?" (Down.)
