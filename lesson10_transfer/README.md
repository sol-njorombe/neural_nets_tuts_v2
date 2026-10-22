# Lesson 10 — Transfer learning

**Read [THEORY.md](THEORY.md) first.**

## What you build

The pieces of a transfer-learning workflow in PyTorch: a few-shot data split,
freezing, a new head on a pretrained backbone, cached feature extraction, and
fine-tuning with two learning rates. A given experiment then compares training
from scratch with starting from a pretrained backbone, on the same handful of
labelled images.

## Setup

Needs PyTorch (see lesson 9's README).

```bash
pytest lesson10_transfer -x
python lesson10_transfer/transfer.py      # after task 4: the three-way comparison
```

`pretrain_backbone` (given) trains a backbone on the digit task and discards its
head. `parity_experiment` (given) runs the comparison in THEORY §3.

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `few_shot_split` | §3 |
| 2 | `freeze`, `trainable_parameters` | §4 |
| 3 | `TransferModel`, `extract_features` | §4 |
| 4 | `fine_tune_optimiser` | §5 |

## Experiments

1. **How few labels?** Run `parity_experiment` with `per_class` = 2, 5, 10, 25, 100.
   At what point does training from scratch catch up?
2. **Features once.** Use `extract_features` to compute the target images'
   features once, then train an `nn.Linear(32, 2)` on them directly. Check that it
   matches the frozen-backbone result, and time both.
3. **When transfer fails.** Pretrain only on images of digits 0–4, then do
   few-shot classification of digits 5–9 (labels `y − 5`) with 5 per class.
   Compare scratch and frozen (THEORY §6 has measured numbers).
4. **Fine-tune too hard.** In `parity_experiment`, skip the frozen phase and
   fine-tune straight away with `backbone_lr=0.01`. What happens to accuracy?
5. **The real thing** (optional, downloads about 100 MB). Follow THEORY §7 with
   `torchvision` on a few dozen of your own photos in two folders, for example
   "keep" and "reject".

## The limit

This is the last technique lesson. What remains is applying everything to a
problem of your own, where most of the work is not the network at all: getting
the data, splitting it honestly, choosing a metric, and comparing against a
simple baseline. Lesson 11.

<details>
<summary>Hints (open only after trying)</summary>

- **few_shot_split:** `np.flatnonzero(labels == c)` gives the indices of class c;
  `np.setdiff1d(np.arange(n), chosen)` gives everything else.
- **TransferModel.forward:** `self.head(self.backbone(X))`.
- **extract_features:** `backbone.eval()` then `with torch.no_grad(): return backbone(X)`.
- **fine_tune_optimiser:** set `requires_grad = True` on every backbone parameter
  before building the optimiser.
</details>
