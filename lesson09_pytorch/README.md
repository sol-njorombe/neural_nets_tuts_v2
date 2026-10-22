# Lesson 9 — PyTorch

**Read [THEORY.md](THEORY.md) first.**

## What you build

Lessons 5 and 6 reimagined in PyTorch:
- Rebuild the XOR network using raw PyTorch tensors, verifying gradients match your Lesson 5 analytical formulas to 12 decimal places.
- Implement a complete digit classifier as an `nn.Module` subclass.
- Transfer hand-trained weights from your Lesson 6 NumPy network into PyTorch's `nn.Linear` layers, proving numerical equivalence.
- Train the model using PyTorch's canonical optimization loop (`zero_grad`, forward, `backward`, `step`).

## Setup

Verify PyTorch installation:

```bash
pip install -e ".[torch]"
pytest lesson09_pytorch -x
python lesson09_pytorch/torch_lesson.py
```

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `to_tensors` | `(X, y) -> (Tensor, Tensor)` | [THEORY §1](THEORY.md#1-a-tensor-is-lesson-3-plus-lesson-8) |
| 2 | `torch_derivative` | `(f, x) -> float` | [THEORY §2](THEORY.md#2-autograd-engine) |
| 3 | `forward`, `train_step` (raw tensors) | XOR with PyTorch autograd | [THEORY §3](THEORY.md#3-lesson-5-xor-network-with-custom-calculus-deleted) |
| 4 | `DigitClassifier`, `count_parameters`, `copy_from_numpy` | `nn.Module` architecture | [THEORY §4](THEORY.md#4-nnmodule-lesson-6s-bricks-in-pytorch) |
| 5 | `train_epoch`, `evaluate` | Standard training workflow | [THEORY §5](THEORY.md#5-the-canonical-pytorch-training-loop) |

### Task 1: `to_tensors`
- Convert input array $X$ to `torch.float32`: `torch.tensor(X, dtype=torch.float32)`.
- Convert target vector $y$ to `torch.int64` (required by `CrossEntropyLoss`): `torch.tensor(y, dtype=torch.long)`.

### Task 2: `torch_derivative`
- Create a 0D float tensor with gradient tracking: `t = torch.tensor(float(x), requires_grad=True)`.
- Evaluate `y = f(t)`.
- Run `y.backward()`.
- Return `float(t.grad.item())`.

### Task 3: Raw Tensor Network
- `forward(X, W1, b1, W2, b2)`:
  - $H = \sigma(X @ W_1 + b_1)$
  - $A = \sigma(H @ W_2 + b_2)$
- `train_step(X, Y, W1, b1, W2, b2, lr)`:
  - Compute MSE loss: `loss = torch.mean((A - Y) ** 2)`.
  - Invoke `loss.backward()`.
  - Update tensors in-place within `with torch.no_grad():`:
    ```python
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
    ```
  - Return `float(loss.item())`.

### Task 4: `DigitClassifier` and Weight Transfer
- Define architecture: `hidden = nn.Linear(64, 32)`, `relu = nn.ReLU()`, `output = nn.Linear(32, 10)`.
- `count_parameters(model)`: count total elements in `p.numel()` across `model.parameters()`.
- `copy_from_numpy(model, dense_hidden, dense_output)`:
  - **Transposition Trap**: Copy weights transposed:
    ```python
    with torch.no_grad():
        model.hidden.weight.copy_(torch.tensor(dense_hidden.W.T))
        model.hidden.bias.copy_(torch.tensor(dense_hidden.b))
        model.output.weight.copy_(torch.tensor(dense_output.W.T))
        model.output.bias.copy_(torch.tensor(dense_output.b))
    ```

### Task 5: Training and Evaluation
- `train_epoch(model, dataloader, optimiser, loss_fn)`:
  - Set `model.train()`.
  - Execute canonical loop: `optimiser.zero_grad()`, `loss = loss_fn(...)`, `loss.backward()`, `optimiser.step()`.
- `evaluate(model, X, y)`:
  - Set `model.eval()`.
  - Inside `with torch.no_grad():`: compute predictions via `model(X).argmax(dim=1)` and compute classification accuracy.

---

## Experiments

1. **Compare Optimizers in PyTorch**:
   Train `DigitClassifier` using:
   - `torch.optim.SGD(model.parameters(), lr=0.1)`
   - `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)`
   - `torch.optim.Adam(model.parameters(), lr=0.001)`
   Compare convergence speed against your scratch implementations from Lesson 7.
2. **Architecture Scaling**:
   Expand the hidden layer width to 256 and add a second hidden layer. Insert `nn.Dropout(0.2)` between layers. Observe the effect on the train/val generalization gap.
3. **Inspect the PyTorch Backward Graph**:
   After evaluating `loss`, print `loss.grad_fn` and trace backward dependencies via `.next_functions`. Contrast this dynamic graph structure with the `Value` DAG built in Lesson 8.
4. **Hardware Acceleration**:
   If running on Apple Silicon or CUDA, dispatch the model and batches to accelerator hardware:
   `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); model.to(device)`.

---

## Self-Check Questions

1. **Abstraction Mapping**: Which earlier lesson's implementation does `loss.backward()` replace? Which does `optimiser.step()` replace?
   *(`loss.backward()` replaces the reverse topological traversal and local chain rule from Lesson 8; `optimiser.step()` replaces the parameter update rules from Lesson 7).*
2. **Gradient Zeroing**: What happens if `optimiser.zero_grad()` is omitted from a training loop?
   *(Gradients from successive minibatches sum together indefinitely, causing the effective step size to grow uncontrollably and destabilizing training).*
3. **Logit Cross-Entropy**: Why does `nn.CrossEntropyLoss` expect unnormalized logits rather than softmax probabilities?
   *(For numerical stability: it combines $\log$ and $\text{softmax}$ into a single fused LogSumExp kernel, preventing underflow and overflow).*
4. **Weight Storage Layout**: Why does PyTorch store linear weights as `(d_out, d_in)` while our scratch `Dense` stored them as `(d_in, d_out)`?
   *(PyTorch implements $Y = X W^T + b$ so that row $i$ of the weight matrix directly represents the incoming weight vector for output neuron $i$).*
5. **Autograd Context Management**: Why must parameter updates take place inside `with torch.no_grad():`?
   *(Because in-place updates are not differentiable mathematical operations in the forward model; without `torch.no_grad()`, PyTorch raises a runtime error regarding in-place leaf modification).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | Updated parameter tensor outside `torch.no_grad()` | Wrap manual updates inside `with torch.no_grad():` |
| Loss decreases initially, then suddenly explodes | Forgot to call `optimiser.zero_grad()` | Ensure `optimiser.zero_grad()` precedes every `loss.backward()` |
| `RuntimeError: expected scalar type Long but found Int` | Target label tensor was not cast to 64-bit integer | Use `dtype=torch.long` in `torch.tensor(y, dtype=torch.long)` |
| `copy_from_numpy` raises shape mismatch error | Forgot to transpose weights when loading into PyTorch | Transpose weights: `torch.tensor(dense.W.T)` |
| Tests skipped during pytest run | PyTorch extra is not installed in the environment | Install torch: `pip install -e ".[torch]"` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`to_tensors`**:
  ```python
  return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
  ```
- **`train_step`**:
  Compute `A` and `loss` outside `torch.no_grad()`. Perform updates and zeroing inside `with torch.no_grad():`.
- **`copy_from_numpy`**:
  ```python
  with torch.no_grad():
      model.hidden.weight.copy_(torch.tensor(dense_hidden.W.T))
      model.hidden.bias.copy_(torch.tensor(dense_hidden.b))
      model.output.weight.copy_(torch.tensor(dense_output.W.T))
      model.output.bias.copy_(torch.tensor(dense_output.b))
  ```
- **`evaluate`**:
  ```python
  model.eval()
  with torch.no_grad():
      preds = model(X).argmax(dim=1)
      return float((preds == y).float().mean().item())
  ```
</details>
