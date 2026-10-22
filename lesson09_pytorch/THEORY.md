# Lesson 9 — Theory: PyTorch

You have now built, by hand, every core principle PyTorch is made of. This lesson is mostly
translation: Lessons 5, 6, and 8 revisited in the production tool you use for real deep learning systems.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Map Course Concepts to PyTorch**: Map every scratch abstraction (`Value`, `Layer`, `Dense`, `Sequential`, `sgd_step`) directly to its PyTorch equivalent (`Tensor`, `nn.Module`, `nn.Linear`, `nn.Sequential`, `torch.optim`).
2. **Execute the Standard Training Loop**: Write and explain the canonical 4-line PyTorch training step (`zero_grad`, forward loss, `backward`, `step`).
3. **Control Autograd Contexts**: Explain why parameter updates must occur inside `with torch.no_grad():` and why optimizer gradients must be zeroed.
4. **Navigate Weight Layout Transpositions**: Reconcile PyTorch's `nn.Linear` transposed weight layout $(d_{\text{out}}, d_{\text{in}})$ with our scratch $(d_{\text{in}}, d_{\text{out}})$ conventions.
5. **Appreciate Framework Abstractions**: Identify what PyTorch handles automatically (accelerator dispatch, dynamic graph caching, C++ kernels) and recognize when to apply scratch principles for debugging.

---

## 1. A tensor is lesson 3 plus lesson 8

- Lesson 3's NumPy arrays are fast and vectorized, but discard computational provenance.
- Lesson 8's `Value` nodes track provenance and autograd, but operate on individual Python scalars.

A PyTorch **Tensor** fuses both concepts: a contiguous multi-dimensional array with native GPU acceleration that keeps an automatic differentiation receipt graph.
Constructing a tensor with `requires_grad=True` instructs the C++ autograd engine to record every operation. Invoking `.backward()` on a scalar output computes gradients across all ancestor leaf tensors, storing results in `.grad`.

### Key Differences Between NumPy and PyTorch

| Feature | NumPy Array | PyTorch Tensor | Reason / Caveat |
|---|---|---|---|
| **Dimension index** | `axis=0` | `dim=0` | Syntax convention |
| **Default Float** | `float64` (double) | **`float32`** (single) | Halves memory bandwidth; optimized on GPU tensor cores |
| **Class Labels** | Any integer type | **`int64` (`torch.long`)** | Mandatory for `nn.CrossEntropyLoss` |
| **Scalar extraction** | `float(arr)` | `tensor.item()` | Extracts native Python float from single-element tensor |
| **Hardware** | CPU only | CPU, CUDA (`cuda`), Apple Silicon (`mps`) | Seamless tensor dispatch via `.to(device)` |

---

## 2. Autograd Engine

```python
x = torch.tensor(3.0, requires_grad=True)
y = x * x
y.backward()
print(x.grad)  # tensor(6.0)
```

Three foundational principles carry over directly from Lesson 8:

1. **Gradients Accumulate via `+=`**: Successive `.backward()` calls sum additively into `.grad`. You must explicitly reset gradients via `p.grad.zero_()` or `optimiser.zero_grad()` before each optimization step.
2. **Parameter Updates Must Bypass the Graph**: Updating parameters is an optimization decision, not a differentiable mathematical model transformation. In-place parameter modification must take place within `with torch.no_grad():` to avoid contaminating the computational DAG.
3. **Leaf Tensors**: Only user-created tensors with `requires_grad=True` retain `.grad` buffers after backpropagation. Intermediate activation nodes release their gradients automatically to conserve VRAM.

---

## 3. Lesson 5 XOR Network, with Custom Calculus Deleted

Rebuilding the Lesson 5 XOR network using raw PyTorch tensors:

```python
H = torch.sigmoid(X @ W1 + b1)          # Exact same forward algebra
A = torch.sigmoid(H @ W2 + b2)
loss = ((A - Y) ** 2).mean()
loss.backward()                          # Replaces dA, dZ2, dW2, db2, dH, dZ1, dW1, db1
```

Testing PyTorch's computed gradients against Lesson 5's hand-derived analytical formulas yields numerical agreement to 12 decimal places. They are the exact same multivariable chain rule, computed automatically.

---

## 4. `nn.Module`: Lesson 6's Bricks in PyTorch

Every structural component written in Lesson 6 maps 1-to-1 to PyTorch:

| Built by Hand (Lesson 6) | PyTorch Equivalent | Responsibility |
|---|---|---|
| `Layer` abstract base | `nn.Module` | Base interface for model components |
| `Dense(d_in, d_out)` | `nn.Linear(d_in, d_out)` | Affine projection with learned parameters |
| `ReLU()`, `Sigmoid()` | `nn.ReLU()`, `nn.Sigmoid()` | Non-linear activation layers |
| `Sequential([...])` | `nn.Sequential(...)` | Sequential container module |
| `model.params()` | `model.parameters()` | Generator of learnable tensor parameters |
| `layer.backward(dY)` | Built-in Autograd | Handled transparently by backward C++ graph |
| `he_normal` (Lesson 7) | `init.kaiming_normal_` | Variance-calibrated parameter initialization |

### The Layout Trap: Transposed Weights in `nn.Linear`

`nn.Linear(d_in, d_out)` allocates its weight parameter with shape **`(d_out, d_in)`**, computing:

$$Y = X W^T + b$$

In our scratch implementation, `Dense` stored $W$ with shape `(d_in, d_out)` and computed $X W + b$.
The underlying linear algebra is completely identical, but when loading numpy weights into PyTorch:
```python
# Must transpose when copying from scratch Dense weights:
model.hidden.weight.copy_(torch.tensor(dense.W.T))
```

---

## 5. The Canonical PyTorch Training Loop

| Built by Hand (Lessons 6–7) | PyTorch Equivalent | Function |
|---|---|---|
| `minibatches(X, y, 32, rng)` | `DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)` | Multithreaded batching and shuffling |
| `softmax` + `cross_entropy` | `nn.CrossEntropyLoss()` | Fused log-sum-exp cross-entropy taking raw **logits** |
| `Adam` (Lesson 7) | `torch.optim.Adam(model.parameters(), lr=1e-3)` | Adaptive moment estimation |
| `EarlyStopping` (Lesson 7) | PyTorch Lightning / Custom hook | Checkpoint saving and validation monitoring |

### The Universal 4-Line Optimization Step

Every supervised deep learning model in PyTorch executes this exact sequence per minibatch:

```python
optimiser.zero_grad()                   # 1. Clear previous gradient accumulations
loss = loss_fn(model(X_batch), y_batch) # 2. Forward pass and loss evaluation
loss.backward()                         # 3. Dynamic reverse-mode backpropagation
optimiser.step()                        # 4. Parameter update via optimizer rule
```

### `model.train()` vs `model.eval()`
Calling `model.eval()` disables training-specific behaviors like Dropout random deactivations and switches BatchNorm to running population statistics. Always set `model.eval()` before evaluating validation metrics.

---

## 6. What PyTorch Adds Beyond Scratch Code

1. **Hardware Acceleration**: Transferring computations to GPU or Apple Silicon takes one method call: `model.to("cuda")` or `model.to("mps")`.
2. **Comprehensive Architecture Zoo**: Hundreds of pre-built layers (Convolutions, Multi-Head Attention, LayerNorm, LSTMs).
3. **Serialization & Checkpointing**: `torch.save(model.state_dict(), "checkpoint.pt")`.
4. **Pretrained Model Ecosystem**: HuggingFace and torchvision model hubs, enabling transfer learning (Lesson 10).
