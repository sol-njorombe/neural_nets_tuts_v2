# PyTorch's own type hints leave a few parameters untyped; that is not your code's problem.
# pyright: reportUnknownMemberType=false
"""Lesson 9 - PyTorch: everything you built, industrial strength.

Read THEORY.md first. Install PyTorch (see README), then fill in each TODO and run:

    pytest lesson09_pytorch -x
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from nnlib.data import load_digits
from nnlib.layers import Dense
from nnlib.types import FloatArray, IntArray


# --- Task 1: tensors (THEORY §1) ---------------------------------------------
def to_tensors(X: FloatArray, y: IntArray) -> tuple[Tensor, Tensor]:
    """NumPy arrays to PyTorch tensors.

    X -> float32 tensor (PyTorch's default float type for networks)
    y -> int64 tensor   (class labels, as cross-entropy expects)
    Use torch.tensor(array, dtype=...).
    """
    # TODO
    # BEGIN SOLUTION
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
    # END SOLUTION


# --- Task 2: autograd (THEORY §2) --------------------------------------------
def torch_derivative(f: Callable[[Tensor], Tensor], x: float) -> float:
    """df/dx at x, using PyTorch's autograd: your lesson 8 engine, for tensors.

        1. x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        2. y = f(x_t)
        3. y.backward()
        4. return x_t.grad as a float
    """
    # TODO
    # BEGIN SOLUTION
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = f(x_t)
    y.backward()
    assert x_t.grad is not None
    return float(x_t.grad)
    # END SOLUTION


# --- Task 3: lesson 5's network with raw tensors (THEORY §3) -----------------
@dataclass
class TwoLayerTensors:
    """GIVEN. Lesson 5's parameters as tensors that record their own gradients."""

    W1: Tensor  # (d_in, d_hidden)
    b1: Tensor  # (d_hidden,)
    W2: Tensor  # (d_hidden, 1)
    b2: Tensor  # (1,)

    @classmethod
    def random(cls, d_in: int, d_hidden: int, rng: np.random.Generator) -> "TwoLayerTensors":
        """Same draws as lesson 5's TwoLayerParams.random, in float64 for exact comparison."""

        def t(a: FloatArray) -> Tensor:
            return torch.tensor(a, dtype=torch.float64, requires_grad=True)

        return cls(
            W1=t(rng.uniform(-1, 1, (d_in, d_hidden))),
            b1=t(rng.uniform(-1, 1, d_hidden)),
            W2=t(rng.uniform(-1, 1, (d_hidden, 1))),
            b2=t(rng.uniform(-1, 1, 1)),
        )

    def all(self) -> list[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def forward(params: TwoLayerTensors, X: Tensor) -> Tensor:
    """Lesson 5's forward pass. Same code, PyTorch spelling: `@` works, sigmoid is torch.sigmoid."""
    # TODO
    # BEGIN SOLUTION
    H = torch.sigmoid(X @ params.W1 + params.b1)
    return torch.sigmoid(H @ params.W2 + params.b2)
    # END SOLUTION


def train_step(params: TwoLayerTensors, X: Tensor, Y: Tensor, lr: float) -> float:
    """One step of gradient descent. Return the loss (before the step) as a float.

        loss = mean squared error of forward(params, X) against Y
        loss.backward()                         # fills every p.grad
        with torch.no_grad():                   # updates are not part of the graph
            for p in params.all():
                p -= lr * p.grad
                p.grad.zero_()                  # lesson 8: gradients accumulate
    (Use loss.item() to get a plain float out of a one-element tensor.)

    Notice what you are NOT writing: dA, dZ2, dH, dZ1...
    """
    # TODO
    # BEGIN SOLUTION
    loss = ((forward(params, X) - Y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for p in params.all():
            grad = p.grad
            assert grad is not None
            p -= lr * grad
            grad.zero_()
    return loss.item()
    # END SOLUTION


# --- Task 4: nn.Module (THEORY §4) -------------------------------------------
class DigitClassifier(nn.Module):
    """Lesson 6's digit network: Linear(64, 32) → ReLU → Linear(32, 10), returning logits."""

    def __init__(self) -> None:
        super().__init__()
        # TODO: create self.hidden, self.relu, self.output as nn.Linear / nn.ReLU / nn.Linear
        # BEGIN SOLUTION
        self.hidden = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)
        # END SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        """(n, 64) -> (n, 10) logits. No softmax: the loss does that (lesson 6 §4)."""
        # TODO
        # BEGIN SOLUTION
        return self.output(self.relu(self.hidden(X)))
        # END SOLUTION


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable numbers: sum of p.numel() over model.parameters()."""
    # TODO
    # BEGIN SOLUTION
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # END SOLUTION


def copy_from_numpy(model: DigitClassifier, hidden: Dense, output: Dense) -> None:
    """Load weights from your lesson 6 Dense layers into the PyTorch model.

    Careful: nn.Linear stores weight as (d_out, d_in) and computes X @ weight.T + bias.
    Your Dense stores W as (d_in, d_out). So weight = W.T.
    Copy in place under torch.no_grad(): model.hidden.weight.copy_(torch.tensor(...)).
    """
    # TODO
    # BEGIN SOLUTION
    with torch.no_grad():
        model.hidden.weight.copy_(torch.tensor(hidden.W.T))
        model.hidden.bias.copy_(torch.tensor(hidden.b))
        model.output.weight.copy_(torch.tensor(output.W.T))
        model.output.bias.copy_(torch.tensor(output.b))
    # END SOLUTION


# --- Task 5: the standard training loop (THEORY §5) --------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """One pass over the loader. Return the mean of the batch losses.

    The five lines every PyTorch project has, per batch:
        optimiser.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimiser.step()
        (record loss.item())
    Call model.train() once at the start.
    """
    # TODO
    # BEGIN SOLUTION
    model.train()
    losses: list[float] = []
    for X_batch, y_batch in loader:
        optimiser.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)
    # END SOLUTION


def evaluate(model: nn.Module, X: Tensor, y: Tensor) -> float:
    """Accuracy, with model.eval() and no gradient tracking (torch.no_grad())."""
    # TODO: argmax over dim=1
    # BEGIN SOLUTION
    model.eval()
    with torch.no_grad():
        predicted = model(X).argmax(dim=1)
    return float((predicted == y).float().mean())
    # END SOLUTION


def make_loader(X: Tensor, y: Tensor, batch_size: int, seed: int) -> DataLoader[tuple[Tensor, ...]]:
    """GIVEN. Lesson 6's minibatches(): shuffled batches, reproducible with a seed."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, generator=generator)


if __name__ == "__main__":
    torch.manual_seed(0)
    d = load_digits()
    X_train, y_train = to_tensors(d.X_train, d.y_train)
    X_val, y_val = to_tensors(d.X_val, d.y_val)
    model = DigitClassifier()
    print(model, "\nparameters:", count_parameters(model))
    loader = make_loader(X_train, y_train, batch_size=32, seed=0)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        loss = train_epoch(model, loader, optimiser, loss_fn)
        print(f"epoch {epoch:2d}  loss {loss:.3f}  val acc {evaluate(model, X_val, y_val):.3f}")
