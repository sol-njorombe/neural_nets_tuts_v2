# Lesson 1 — One neuron

**Read [THEORY.md](THEORY.md) first.**

## What you build

A `Neuron` class that turns three inputs into an output between 0 and 1, plus
a function that scores it against a dataset. Then you act as the learning
algorithm yourself, choosing weights by hand that make the neuron compute OR
and AND.

## Setup

```bash
pytest lesson01_neuron -x
python lesson01_neuron/neuron.py     # after task 4: a random neuron guessing at XOR
```

## The data

Each dataset is a `list[Example]`. An `Example` is a frozen dataclass with:

- `x: tuple[float, ...]`: the inputs, e.g. `(1.0, 0.0, 1.0)`;
- `y: int`: the target, 0 or 1.

Four datasets are given, each listing all 8 combinations of three 0/1 inputs:

| Name | Target is 1 when… |
|---|---|
| `FIRST_INPUT` | the first input is 1 (the other two are distractions) |
| `OR3` | at least one input is 1 |
| `AND3` | all three inputs are 1 |
| `XOR3` | an odd number of inputs are 1 |

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `sigmoid(z)`, safe for any `z` | §3 |
| 2 | `Neuron.random(d_in, rng)` | §5 |
| 3 | `weighted_sum`, `forward`, `predict` | §2 |
| 4 | `accuracy(neuron, data)` | — |
| 5 | `HAND_OR`, `HAND_AND`: choose the weights yourself | §4 |

## Experiments

1. Run the demo. Change the seed a few times. Does any random neuron get XOR3 right?
2. Find the *smallest* positive weights for which `HAND_OR` still gets 100%
   accuracy (keep all three weights equal). What happens to the confidence test?
3. Make a neuron that outputs 1 when **at least two** inputs are on. Then one for
   "the first input is on and the second is off".
4. Try to hand-pick a neuron for `XOR3`. Spend five minutes at most. Lesson 2
   explains why you could not.

## The limit

A random neuron is confidently wrong, and you could fix it by hand only because
there were four numbers to choose and you knew the answer in advance. A network
that reads handwriting has tens of thousands of weights, and nobody knows what
they should be.

The neuron needs to find its own weights. For that it first needs a single
number that says how wrong it is, and a way to improve that number. That is
lesson 2.

<details>
<summary>Hints (open only after trying)</summary>

- **Task 1:** two branches on the sign of `z`. THEORY §3 gives both formulas.
- **Task 2:** a list comprehension over `range(d_in)`; `float(rng.uniform(-1, 1))`.
- **Task 3:** check the lengths first, then start the sum at `self.b` and add
  `w_i * x_i` for each pair from `zip(self.w, x)`. `forward` and `predict` are
  one line each and should call the method before them.
- **Task 4:** count the examples where `neuron.predict(example.x) == example.y`,
  then divide by `len(data)`.
- **Task 5:** with all weights 10, the weighted sums are 0, 10, 20 or 30. For AND
  only 30 may clear the hurdle.
</details>
