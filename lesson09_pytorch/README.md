# Lesson 9 — PyTorch

**Read [THEORY.md](THEORY.md) first.**

## What you build

Lessons 5 and 6 again, in PyTorch: the XOR network with raw tensors (checked
number for number against your lesson 5 gradients), then the digit classifier as
an `nn.Module` trained with the standard PyTorch loop. Along the way you load
your own lesson 6 weights into PyTorch and get identical outputs.

## Setup

Install PyTorch from the repository root:

```bash
pip install -e ".[torch]"        # or: uv sync --extra torch
```

On Linux without a GPU, the CPU-only build is much smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```bash
pytest lesson09_pytorch -x
python lesson09_pytorch/torch_lesson.py
```

If PyTorch is not installed, this lesson's tests are skipped rather than failed.

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `to_tensors` | §1 |
| 2 | `torch_derivative` | §2 |
| 3 | `forward`, `train_step`: lesson 5 with no backward pass | §3 |
| 4 | `DigitClassifier`, `count_parameters`, `copy_from_numpy` | §4 |
| 5 | `train_epoch`, `evaluate` | §5 |

## Experiments

1. **Optimisers.** Swap `Adam` for `torch.optim.SGD(model.parameters(), lr=0.1)`
   and for `SGD(..., lr=0.1, momentum=0.9)`. Compare with your lesson 7 results.
2. **Bigger model, one line.** Make the hidden layer 256 wide and add a second
   one. Add `nn.Dropout(0.2)` after each ReLU. Remember `model.train()` and
   `model.eval()`.
3. **Inspect the graph.** After `loss = ...`, print `loss.grad_fn` and follow
   `.next_functions` a few steps. Compare with your lesson 8 `children` and `op`.
4. **A GPU, if you have one.** On Apple Silicon: `device = torch.device("mps")`,
   then `model.to(device)` and move each batch with `.to(device)`. On a dataset
   this small it may not be faster. Why not?

## The limit

Training from scratch needs a lot of labelled data. The digits are tiny 8×8
images; a model for real photos trained from nothing would need hundreds of
thousands of labelled examples. Lesson 10 starts from a model someone else
already trained and adapts it with very little data. That is how most real
projects begin.

<details>
<summary>Hints (open only after trying)</summary>

- **to_tensors:** `torch.tensor(X, dtype=torch.float32)`.
- **train_step:** compute the loss outside `torch.no_grad()`; only the updates go
  inside it. `loss.item()` gives a float.
- **DigitClassifier.forward:** `self.output(self.relu(self.hidden(X)))`.
- **copy_from_numpy:** `model.hidden.weight.copy_(torch.tensor(hidden.W.T))`, and
  the same for the bias (no transpose) and the output layer.
- **evaluate:** `model(X).argmax(dim=1)` compared with `y`, then `.float().mean().item()`.
</details>
