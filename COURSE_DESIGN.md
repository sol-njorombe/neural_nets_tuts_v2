# Course design

The rules every lesson in this repository follows. When adding or changing a
lesson, check it against this page.

## Goal and audience

- **Goal:** understand every line well enough to teach it. Finishing is not the goal.
- **Audience:** a developer comfortable in Python, with rusty calculus and no
  machine-learning background.
- **Two modes:** the repo must work for someone learning alone, and for someone
  teaching it to another person.

## Rules

### Lesson structure

1. **One new idea per lesson.** If a lesson needs two, split it.
2. **Every lesson folder has the same files:**

   | File | Purpose |
   |---|---|
   | `THEORY.md` | Learning objectives → concept → analogy (only where it helps) → math → worked example with small numbers you can check by hand → diagrams |
   | `README.md` | Setup, tasks, commands to run, experiments, self-check questions, common mistakes & troubleshooting, hints, and the limit |
   | `<name>.py` | The learner's file: typed stubs with TODOs |
   | `test_<name>.py` | The specification, as tests |

   The reference solution lives in `solutions/<lesson>/<name>.py`. The capstone
   (lesson 11) is the one exception: it has no code file or tests, and its
   `README.md` holds a review rubric instead.
3. **Every lesson ends with its limit:** the problem that motivates the next lesson.
4. **Lessons are self-contained.** Each starts from working reference code for
   everything before it, so a mistake in one lesson never carries into the next.
   Finished building blocks live in the `nnlib` package; later lessons import
   from it rather than from the learner's earlier files.

### Code

5. **Typed throughout.** Python 3.12+, full type hints, checked by pyright in
   strict mode. Plain dicts are never used as data structures: anything with a
   fixed set of fields is a dataclass or a class.
6. **Every array is annotated twice:** a type hint and a shape comment on the
   same line, because type hints cannot check shapes.

   ```python
   W: FloatArray  # (d_in, d_out)
   ```
7. **One shape convention for the whole course:** rows are examples.

   ```
   X (n, d_in) @ W (d_in, d_out) + b (d_out,)  →  (n, d_out)
   ```
   This is the convention PyTorch uses, so nothing is relearned at the end.
8. **Gradient names:** `dW`, `db`, `dX` always mean "∂L / ∂ that thing". A
   gradient always has the same shape as the thing it belongs to.
9. **Per-neuron objects appear only in the scalar lessons (0–2).** From the first
   network onward, the unit of code is a `Layer` with `forward` and `backward`.
10. **Randomness is explicit:** a `np.random.Generator` is passed in, never
    created inside a function, so every result is reproducible.

### Math

11. **One notation for the whole course,** defined in [NOTATION.md](NOTATION.md).
    A symbol never changes meaning.
12. **Every formula appears three ways, side by side:** math, shapes, and code.
13. **Every gradient the learner writes is checked** against the numerical
    "wiggle" method from lesson 0, in the tests.

### Tests

14. **Tests follow the task order.** Test names read as a specification; failure
    messages name the likely mistake.
15. **Every function that returns an array has a shape test.**
16. **Each lesson has a payoff test** showing what the lesson made possible.
17. **`pytest --solutions` runs every test against the reference solutions.** CI
    runs it on every push, along with pyright and the stub-sync check.

### Frameworks

18. **NumPy first, PyTorch last.** PyTorch arrives only after the learner has built
    autograd, with a table mapping each PyTorch call to code they wrote.

## Lesson map

| # | Folder | Idea | Payoff | Status |
|---|---|---|---|---|
| 0 | `lesson00_derivatives` | Derivatives, partial derivatives, chain rule | Backprop on a one-weight network, by hand | ✅ |
| 1 | `lesson01_neuron` | One neuron as a typed dataclass | You set weights by hand that compute OR and AND | ✅ |
| 2 | `lesson02_gradient_descent` | Loss and gradient descent | The neuron learns OR, and provably cannot learn XOR | ✅ |
| 3 | `lesson03_shapes` | NumPy and shapes bootcamp: `@`, broadcasting, transpose, reshape, axis sums | Shape fluency, no neural nets | ✅ |
| 4 | `lesson04_batched_neuron` | One neuron on a whole batch | `dW = X.T @ δ`, derived from shapes | ✅ |
| 5 | `lesson05_backprop` | Hidden layer and backpropagation in matrices | XOR solved | ✅ |
| 6 | `lesson06_layers` | Layers, softmax, minibatches | Your code reads handwritten digits | ✅ |
| 7 | `lesson07_generalisation` | Validation, overfitting, initialisation, learning rate, momentum, Adam | You can diagnose a bad training run | ✅ |
| 8 | `lesson08_autograd` | Automatic differentiation | Networks with no hand-written backward pass | ✅ |
| 9 | `lesson09_pytorch` | PyTorch | Lesson 6 in PyTorch, gradients matched to yours | ✅ |
| 10 | `lesson10_transfer` | Transfer learning | A pretrained model plus a small trained head | ✅ |
| 11 | `lesson11_capstone` | Capstone | Your own project | ✅ |

## Maintaining the course

The reference solutions are the source of truth. Learner stubs are generated
from them.

- In `solutions/<lesson>/<name>.py`, wrap each answer in markers:

  ```python
  def forward(self, x: Inputs) -> float:
      # TODO: σ(z)
      # BEGIN SOLUTION
      return sigmoid(self.weighted_sum(x))
      # END SOLUTION
  ```
  In the stub, the block becomes `raise NotImplementedError("TODO: implement forward()")`.
  For a custom stub line, write `# BEGIN SOLUTION stub: HAND_OR: Neuron | None = None`.
- Regenerate stubs with `python tools/make_stubs.py`.
- Before pushing, run what CI runs:

  ```bash
  python tools/make_stubs.py --check   # stubs match solutions
  pyright                              # strict types
  pytest --solutions                   # every test passes on the reference
  ```
- Files with no solution in them (tests, Markdown, given helpers such as
  `lesson08_autograd/tiny_nn.py`) are written directly in the lesson folder.
- `nnlib/` holds finished versions of code that later lessons build on. When a
  lesson's solution changes, update the matching `nnlib` module too.
- Module names must be unique across all lesson folders, because every lesson
  folder is on the import path at once.
