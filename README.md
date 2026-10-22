# Neural Nets From Scratch

A hands-on course that builds neural networks from a single derivative up to
PyTorch. You write every piece yourself first, hit its limit, and only then
reach for the tool that removes that limit.

Each lesson pairs a theory note with a coding puzzle and a test suite, so you
always know whether you have understood something: the tests tell you.

## Why learn this when LLMs exist?

**Build or reuse.** Reuse an existing model when the problem is language,
open-ended reasoning, or messy unstructured input. Train your own when:

- the data is yours and is numeric, tabular, sensor, audio or image data;
- you need answers in milliseconds, on a phone, or at near-zero cost each;
- the task is narrow and high-volume: classify, rank, score, detect anomalies;
- you want to fine-tune or adapt a model rather than only prompt it.

Even if you mostly reuse models, the internals pay off. Embeddings, fine-tuning,
learning rates, overfitting and "why did the loss go to NaN" all stop being
folklore once you have built them.

**What neural nets are good at.** Learning a function from examples when nobody
can write the rules by hand: perception, pattern recognition, prediction from
high-dimensional data. They are a poor fit for tiny datasets, for problems with
exact rules, and for anything that needs a guarantee.

**How the terms relate.**

| Term | What it is |
|---|---|
| Neural network | A function with adjustable numbers (weights), tuned by gradient descent |
| Deep learning | Neural networks with many layers |
| Supervised learning | Learning from examples that come with the right answer. This whole course. |
| Reinforcement learning | Learning from rewards by trial and error. A different *problem*, not a different kind of network. |
| Transfer learning | Starting from a model someone else trained, and adapting it |
| LLM | A very large network (a transformer) trained to predict the next token |

**A short history, and why the course starts with XOR.**

- 1958: Rosenblatt's perceptron, a single trainable neuron. Lessons 1–2.
- 1969: Minsky and Papert show a single-layer perceptron cannot compute XOR. You reproduce this in lesson 2.
- 1986: Rumelhart, Hinton and Williams popularise backpropagation, which trains hidden layers and solves XOR. Lesson 5.
- 2012: AlexNet wins ImageNet with deep networks on GPUs. Deep learning takes over computer vision.
- 2013–2016: DeepMind puts deep networks inside reinforcement learning agents: Atari, then AlphaGo.
- 2017: The transformer, the architecture behind every current LLM.
- 2020: AlphaFold 2 predicts protein structures at near-experimental accuracy (supervised deep learning, not RL).
- 2024: Nobel Prizes for Hopfield and Hinton (physics) and for Hassabis and Jumper (chemistry).

Every item on that list runs on the loop you write in lesson 2: predict, measure
the error, compute slopes, nudge the weights.

## Setup

Work locally: the loop of "edit a file, run its tests" is awkward in notebooks.

```bash
git clone <this repo> && cd nn-from-scratch
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"           # numpy, pytest, pyright
```

The editable install also makes `nnlib` importable: a small library holding
finished versions of each lesson's code, which later lessons build on. Lessons 9
and 10 need PyTorch: `pip install -e ".[torch]"` (or `uv sync --extra torch`).

Use an editor with Python type checking (VS Code with Pylance, or PyCharm).
The type hints are there to answer "what is this variable?" as you type.

## How each lesson works

1. **Read `THEORY.md`.** The concept, the math, and a worked example with real numbers.
2. **Read `README.md`.** What you are building and the tasks, in order.
3. **Fill in the TODOs** in the lesson's `.py` file, running its tests as you go:

   ```bash
   pytest lesson00_derivatives -x            # stop at the first failure
   pytest lesson00_derivatives -k chain      # only tests with "chain" in the name
   ```
4. **Do the experiments** in the README. This is where intuition comes from.

**Stuck?** In order: re-read the docstring and the test that fails, open the
hints at the bottom of the README, then look at `solutions/`.
`pytest --solutions` runs the tests against the reference answers.

**Teaching someone?** Each lesson's `TEACHING.md` has learning objectives,
discussion questions, and the mistakes learners usually make.

## Lessons

| # | Folder | You learn | Status |
|---|---|---|---|
| 0 | `lesson00_derivatives` | Derivatives, partial derivatives, the chain rule, in Python | ✅ |
| 1 | `lesson01_neuron` | One neuron as a typed class; the forward pass | ✅ |
| 2 | `lesson02_gradient_descent` | Loss and gradient descent; the XOR wall | ✅ |
| 3 | `lesson03_shapes` | NumPy and matrix shapes | ✅ |
| 4 | `lesson04_batched_neuron` | One neuron on a whole batch | ✅ |
| 5 | `lesson05_backprop` | Hidden layers and backpropagation; XOR solved | ✅ |
| 6 | `lesson06_layers` | Layers, softmax, minibatches; reading handwritten digits | ✅ |
| 7 | `lesson07_generalisation` | Overfitting, initialisation, optimisers | ✅ |
| 8 | `lesson08_autograd` | Automatic differentiation | ✅ |
| 9 | `lesson09_pytorch` | PyTorch | ✅ |
| 10 | `lesson10_transfer` | Transfer learning | ✅ |
| 11 | `lesson11_capstone` | A project of your own, with four briefs to choose from | ✅ |

[NOTATION.md](NOTATION.md) defines every symbol. [COURSE_DESIGN.md](COURSE_DESIGN.md)
holds the rules every lesson follows and how to maintain the course.
