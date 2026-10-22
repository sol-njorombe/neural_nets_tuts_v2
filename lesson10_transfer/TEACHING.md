# Lesson 10 — Teaching notes

## Learning objectives

By the end, the learner can:

1. Explain transfer learning with the backbone/head picture, and why pretrained features help.
2. Implement feature extraction with a frozen backbone, and explain why features can be cached.
3. Implement fine-tuning with parameter groups, and explain why the backbone's rate is smaller.
4. Design a fair comparison (different images for pretraining and evaluation).
5. Say when transfer will not help.

## Suggested session (about 1.5 hours)

| Time | Activity |
|---|---|
| 10 min | The photographer analogy; draw backbone and head |
| 40 min | Tasks 1–4 |
| 30 min | Experiments 1 and 3 |
| 10 min | Discussion: connect to embeddings and LLM fine-tuning |

## Discussion questions

1. Why must the pretraining images differ from the evaluation images?
2. The frozen model has 66 trainable parameters; the scratch model about 6,300. With 50 labelled images, which do you expect to overfit, and why?
3. Experiment 3 made transfer worse than scratch. What does that say about choosing a pretrained model?
4. What is an "embedding API" in the language of this lesson?
5. For a project of yours, what pretrained backbone would you start from, and what would the head output?

## Common mistakes

| Symptom | Likely cause |
|---|---|
| Frozen backbone changes during training | Gave `model.parameters()` to the optimiser instead of the trainable ones, and never set `requires_grad = False` |
| `extract_features` output has `requires_grad=True` | Missing `torch.no_grad()` |
| Fine-tuning does nothing to the backbone | Did not set `requires_grad = True` again |
| Few-shot split test fails on overlap | Built train and test from different shuffles |

## Checks for understanding

- "A ResNet backbone outputs 2048 features. You want 12 slider values. Shape of the head's weight in PyTorch?" ((12, 2048).)
- "Name two reasons to train the head before unfreezing the backbone."
