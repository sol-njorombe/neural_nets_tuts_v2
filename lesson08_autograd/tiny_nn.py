"""GIVEN - a complete neural-network library built on YOUR Value class.

Read it once the engine tests pass. Notice what is missing: there is not one
backward() method or hand-derived gradient in this file. You describe the
forward pass; engine.py works out the rest.

    python lesson08_autograd/tiny_nn.py
"""

from itertools import product

import numpy as np

from engine import Value


class Neuron:
    def __init__(self, d_in: int, rng: np.random.Generator, activation: bool = True) -> None:
        self.w = [Value(float(rng.uniform(-1, 1))) for _ in range(d_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list[Value] | tuple[float, ...]) -> Value:
        z = self.b
        for w_i, x_i in zip(self.w, x):
            z = z + w_i * x_i
        return z.tanh() if self.activation else z

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, activation: bool = True) -> None:
        self.neurons = [Neuron(d_in, rng, activation) for _ in range(d_out)]

    def __call__(self, x: list[Value] | tuple[float, ...]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """A multi-layer perceptron. MLP(3, [4, 4, 1], rng) is 3 inputs → 4 → 4 → 1 output."""

    def __init__(self, d_in: int, layer_sizes: list[int], rng: np.random.Generator) -> None:
        sizes = [d_in] + layer_sizes
        last = len(layer_sizes) - 1
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, activation=i < last) for i in range(len(layer_sizes))]

    def __call__(self, x: tuple[float, ...]) -> Value:
        h: list[Value] | tuple[float, ...] = x
        for layer in self.layers:
            h = layer(h)
        assert isinstance(h, list) and len(h) == 1, "this little library assumes one output"
        return h[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def train(model: MLP, data: dict[tuple[float, ...], float], lr: float, epochs: int) -> list[float]:
    """Full-batch gradient descent on mean squared error. Returns the loss history."""
    history: list[float] = []
    for _ in range(epochs):
        # Forward: one big expression for the loss. The graph builds itself.
        total = Value(0.0)
        for x, y in data.items():
            total = total + (model(x) - y) ** 2
        loss = total / len(data)
        history.append(loss.data)

        # Backward: one call, no formulas.
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad
    return history


XOR3: dict[tuple[float, ...], float] = {
    tuple(float(b) for b in bits): float(sum(bits) % 2) for bits in product((0, 1), repeat=3)
}

if __name__ == "__main__":
    # Two hidden layers. In lesson 5 that would have meant deriving new formulas.
    model = MLP(3, [4, 4, 1], np.random.default_rng(1))
    history = train(model, XOR3, lr=0.1, epochs=1000)
    print(f"loss {history[0]:.4f} -> {history[-1]:.6f}")
    for x, y in XOR3.items():
        print(x, "want", y, "got", round(model(x).data, 3))
