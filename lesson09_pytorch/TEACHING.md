# Lesson 9 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Map each PyTorch construct in this lesson to the course code it replaces.
2. Write the standard four-line training step and explain each line.
3. Explain why parameter updates go inside `torch.no_grad()` and why gradients are zeroed.
4. Explain `nn.Linear`'s transposed weight layout.
5. Name what PyTorch adds (GPUs, many layer types, pretrained models) and what it hides.

## Suggested session (about 1.5 hours)

| Time | Activity |
|---|---|
| 10 min | Walk through THEORY §4–5's translation tables together |
| 50 min | Tasks 1–5 |
| 20 min | Experiments 1 and 3 |
| 10 min | Discussion |

## Discussion questions

1. Which lesson's code does `loss.backward()` replace? Which does `optimiser.step()` replace?
2. What would happen without `optimiser.zero_grad()`? Where did you see the same rule before?
3. Why does `nn.CrossEntropyLoss` take logits instead of probabilities? (Numerical stability: lesson 6's subtract-the-max trick inside one function.)
4. The PyTorch test compared gradients to 12 decimal places. What does that agreement tell you about lessons 5 and 8?
5. When training goes wrong in PyTorch, which earlier lesson's tools do you reach for?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | Update outside `torch.no_grad()` |
| Loss decreases then explodes | Forgot to zero gradients between steps |
| `expected scalar type Long` | Labels not int64 |
| `copy_from_numpy` shape error | Forgot to transpose W |
| Tests skipped | PyTorch not installed in the active environment |

## Checks for understanding

- "`nn.Linear(128, 10)`: shape of `weight`, shape of `bias`, number of parameters?" ((10, 128), (10,), 1290.)
- "Put these in order: `loss.backward()`, `optimiser.step()`, `optimiser.zero_grad()`, forward." (zero_grad, forward, backward, step.)
