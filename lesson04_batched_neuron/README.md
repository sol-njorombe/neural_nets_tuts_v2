# Lesson 4 — One neuron on a whole batch

**Read [THEORY.md](THEORY.md) first.**

## What you build

Lesson 2's neuron again, but every example goes through at once as a matrix. The
forward pass is two lines, the gradient is four, and there are no loops except
over epochs. The tests check your gradient against both the wiggle method and
your lesson 2 loop version.

## Setup

```bash
pytest lesson04_batched_neuron -x
python lesson04_batched_neuron/batched.py
```

## The types

| Type | Fields | Shapes |
|---|---|---|
| `NeuronParams` | `W`, `b` | `(d_in, 1)`, `(1,)` |
| `Cache` | `X`, `A` | `(n, d_in)`, `(n, 1)` — saved by forward for backward |
| `Grads` | `dW`, `db` | same shapes as `W` and `b` |

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `forward` and `mse` | §2 |
| 2 | `backward`: four lines | §3 |
| 3 | `train`: full-batch gradient descent | §4, §5 |

## Experiments

1. Add `print(X.shape, cache.A.shape, dZ.shape, dW.shape)` inside your `backward`
   and read each shape aloud with what it means ("8 examples, 1 output").
2. Remove the `/ n` from `dA`. Which tests fail? Then try `lr=1/8` with that
   change. Why does it work again?
3. Change `Y` to a flat array with `Y.reshape(-1)` before calling `train`. What
   happens to the loss, and why does nothing crash?
4. In the speed test, try `epochs=200` on the lesson 2 code with 1,000 examples
   and 20 inputs. Compare the time.

## The limit

The neuron is fast now, but it is still one neuron, and the XOR test still fails
for the reason in lesson 2 §6. The fix is a **hidden layer**: several neurons
whose outputs feed a final neuron. With `W` already a matrix, that is one more
`@` going forward and one more application of the backward table. Lesson 5.

<details>
<summary>Hints (open only after trying)</summary>

- **forward:** `Z = X @ params.W + params.b`, then `sigmoid(Z)`.
- **mse:** wrap in `float(...)`.
- **backward:** `n = cache.X.shape[0]`. Use `*` for the elementwise steps and `@`
  only for `dW`. If `db` has shape `(1, 1)` you used `keepdims=True`; the tests
  want `(1,)`, like `b`.
- **train:** `params.W -= lr * grads.dW` updates the array in place.
</details>
