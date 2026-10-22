# Lesson 9 — Theory: PyTorch

You have now built, by hand, every idea PyTorch is made of. This lesson is mostly
translation: lessons 5, 6 and 8 again, in the tool you would use for real work.

## 1. A tensor is lesson 3 plus lesson 8

- Lesson 3's arrays are fast but forget where they came from.
- Lesson 8's Values remember, but one number at a time.

A PyTorch **tensor** is an array that can keep receipts. Create it with
`requires_grad=True` and every operation on it is recorded; `.backward()` on a
result fills `.grad` on every tensor that fed into it. Your lesson 8 API, almost
word for word.

Tensors look and behave like NumPy arrays: `.shape`, `@`, `*`, broadcasting,
`.sum(dim=0)`, `.T`. Two differences to know:

| NumPy | PyTorch |
|---|---|
| `axis=0` | `dim=0` |
| default float64 | networks use **float32** (half the memory, much faster on GPUs) |
| labels any int type | class labels must be **int64** for cross-entropy |
| `float(x)` for a 1-element array | `x.item()` |

## 2. Autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = x * x
y.backward()
x.grad            # tensor(6.)
```

Three rules carry over directly from lesson 8:

- **Gradients accumulate.** Calling `backward` twice adds. Clear with
  `p.grad.zero_()` or `optimiser.zero_grad()` before each step.
- **Updates must not be recorded.** Changing a parameter is not part of the
  model's maths, so it goes inside `with torch.no_grad():`.
- **Only leaves you asked about get `.grad`.** Tensors created with
  `requires_grad=True` (your parameters) keep their gradients.

## 3. Lesson 5, with the backward pass deleted

Task 3 rebuilds the XOR network with raw tensors:

```python
H = torch.sigmoid(X @ W1 + b1)          # lesson 5's forward, unchanged
A = torch.sigmoid(H @ W2 + b2)
loss = ((A - Y) ** 2).mean()
loss.backward()                          # replaces dA, dZ2, dW2, db2, dH, dZ1, dW1, db1
```

The test compares PyTorch's update with your lesson 5 formulas to 12 decimal
places. They agree because they are the same chain rule, applied by a machine.

## 4. `nn.Module`: lesson 6's bricks

| You wrote (lesson 6) | PyTorch |
|---|---|
| `Layer` base class | `nn.Module` |
| `Dense(64, 32, rng)` | `nn.Linear(64, 32)` |
| `ReLU()`, `Sigmoid()` | `nn.ReLU()`, `nn.Sigmoid()` |
| `Sequential([...])` | `nn.Sequential(...)`, or your own `nn.Module` subclass |
| `model.params()` | `model.parameters()` |
| `layer.backward(dY)` | never written: autograd |
| He initialisation (lesson 7) | Chosen for you (`nn.Linear` uses a similar scaled uniform) |

A custom module creates its layers in `__init__` and says how data flows in
`forward`. Calling `model(X)` runs `forward`.

**One gotcha: weights are stored transposed.** `nn.Linear(d_in, d_out)` stores
`weight` with shape **(d_out, d_in)** and computes `X @ weight.T + bias`. Your
`Dense` stored `W` as (d_in, d_out) and computed `X @ W + b`. Same maths, other
layout. Task 4 loads your lesson 6 weights into a PyTorch model and checks the
outputs match exactly, which only works once you transpose.

## 5. The training loop every PyTorch project has

| You wrote (lessons 6–7) | PyTorch |
|---|---|
| `minibatches(X, y, 32, rng)` | `DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)` |
| `softmax` + `cross_entropy` + `cross_entropy_backward` | `nn.CrossEntropyLoss()`, which takes **logits** |
| `SGD`, `Momentum`, `Adam` (lesson 7) | `torch.optim.SGD`, `torch.optim.SGD(momentum=0.9)`, `torch.optim.Adam` |
| `optimiser.step()` | `optimiser.step()` |
| clearing `.grad` (lesson 8) | `optimiser.zero_grad()` |
| `accuracy(model, X, y)` | `model(X).argmax(dim=1) == y`, inside `torch.no_grad()` |

Per batch:

```python
optimiser.zero_grad()                   # lesson 8: clear accumulated blame
loss = loss_fn(model(X_batch), y_batch) # forward + loss; the graph builds itself
loss.backward()                         # every parameter's .grad
optimiser.step()                        # lesson 7's update rule
```

**`model.train()` and `model.eval()`** switch layers that behave differently
during training, such as dropout (lesson 7 §8). This model has none, but the
habit matters as soon as one does.

## 6. What PyTorch adds that you did not build

- **GPUs.** `model.to("cuda")` or `model.to("mps")` (Apple Silicon) and the same
  code runs many times faster on large models.
- **Hundreds of bricks:** convolutions for images, attention for sequences,
  normalisation layers, embeddings. Each obeys lesson 6's contract; you never
  write their backward.
- **Saving and loading:** `torch.save(model.state_dict(), "model.pt")`.
- **Pretrained models,** which lesson 10 uses.

## 7. The limits of this step

- **It hides everything.** `loss.backward()` and `optimiser.step()` are so short
  they teach nothing, which is why they come last. When training misbehaves you
  debug with lessons 2–7, not with PyTorch.
- **Training from scratch needs data.** 1,437 digit images were enough for 8×8
  digits. A photo classifier trained from scratch would need hundreds of
  thousands of images. Lesson 10 shows how to borrow what someone else's model
  already learned.

**TensorFlow and Keras** offer the same ideas: `tf.GradientTape` is autograd,
`keras.layers.Dense` is `nn.Linear` (stored the lesson 6 way round), and
`model.fit` is the training loop. Having built the ideas, you can read either.
