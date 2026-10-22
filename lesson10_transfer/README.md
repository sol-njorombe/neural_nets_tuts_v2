# Lesson 10 — Transfer learning

**Read [THEORY.md](THEORY.md) first.**

## What you build

A complete transfer-learning engineering pipeline in PyTorch:
- A balanced `few_shot_split` with disjoint test partitioning.
- Parameter freezing (`freeze`) and trainable parameter filtering.
- A composite `TransferModel` coupling a pre-trained backbone to a custom head.
- Pre-computed feature extraction (`extract_features`) with `torch.no_grad()`.
- A differential two-rate fine-tuning optimizer (`fine_tune_optimiser`).

Then you run a rigorous three-way benchmark comparing *From Scratch*, *Frozen Backbone*, and *Fine-Tuned* models across varying label budgets.

## Setup

Verify PyTorch is installed:

```bash
pytest lesson10_transfer -x
python lesson10_transfer/transfer.py
```

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `few_shot_split` | `(X, y, per_class, rng) -> (X_tr, y_tr, X_te, y_te)` | [THEORY §3](THEORY.md#3-the-controlled-transfer-experiment) |
| 2 | `freeze`, `trainable_parameters` | Parameter gradient toggling | [THEORY §4](THEORY.md#4-feature-extraction-freezing-the-backbone) |
| 3 | `TransferModel`, `extract_features` | Architecture and cached extraction | [THEORY §4](THEORY.md#4-feature-extraction-freezing-the-backbone) |
| 4 | `fine_tune_optimiser` | Multi-rate Adam optimizer | [THEORY §5](THEORY.md#5-fine-tuning-differential-learning-rates) |

### Task 1: `few_shot_split`
- For each unique class $c \in \{0, 1, \dots, C-1\}$:
  - Find matching indices using `np.flatnonzero(labels == c)`.
  - Permute indices with `rng.permutation` and select the first `per_class` indices.
- Pool selected indices into training indices; all remaining indices form the test set (`np.setdiff1d(np.arange(n), train_indices)`).
- Return partitioned tensor tuples.

### Task 2: Freezing and Filtering
- `freeze(module)`: iterate through `module.parameters()` setting `p.requires_grad = False`.
- `trainable_parameters(module)`: return a list of parameters where `p.requires_grad` is `True`.

### Task 3: `TransferModel` and `extract_features`
- `TransferModel`:
  - `__init__(backbone, n_features, n_classes)`: store `self.backbone = backbone`, create `self.head = nn.Linear(n_features, n_classes)`.
  - `forward(X)`: return `self.head(self.backbone(X))`.
- `extract_features(backbone, X)`:
  - Set `backbone.eval()`.
  - Inside `with torch.no_grad():`: compute and return `backbone(X)`.

### Task 4: `fine_tune_optimiser`
- Set `p.requires_grad = True` on all parameters in both `model.backbone` and `model.head`.
- Return a `torch.optim.Adam` instance with parameter groups:
  ```python
  return torch.optim.Adam([
      {"params": model.head.parameters(), "lr": head_lr},
      {"params": model.backbone.parameters(), "lr": backbone_lr}
  ])
  ```

---

## Experiments

1. **Label Budget Sweep**:
   Run `parity_experiment` over `per_class` $\in \{2, 5, 10, 25, 100\}$.
   At what training set size does training from scratch begin to match the performance of the frozen backbone?
2. **Speed Comparison via Feature Caching**:
   Extract features once using `extract_features` and train an `nn.Linear(32, 2)` directly on cached tensors. Compare wall-clock execution time against running the full forward-backward graph through the backbone every epoch.
3. **Negative Transfer Demonstration**:
   Pretrain the backbone exclusively on digits $0$ through $4$. Then evaluate few-shot transfer on digits $5$ through $9$. Notice how the frozen backbone underperforms random initialization because learned features fail to transfer.
4. **Catastrophic Forgetting via Aggressive Fine-Tuning**:
   In `parity_experiment`, skip the frozen pre-training stage and fine-tune immediately with a large backbone rate `backbone_lr = 0.01`. What happens to validation accuracy?
5. **Real-World Vision Transfer**:
   Use `torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)` on your own collection of photos, replacing the final FC layer to build a binary photo filter.

---

## Self-Check Questions

1. **Experimental Hygiene**: Why must the pretraining digit images be strictly separated from the target few-shot parity images?
   *(If the backbone were pre-trained on the target images, the backbone would have already memorised their visual structure, creating experimental data leakage).*
2. **Overfitting Dynamics**: The frozen model has 66 learnable parameters, while the scratch model has 6,300 parameters. Why does the scratch model overfit severely when trained on 50 images?
   *(Because 50 images provide far too few constraints to guide 6,300 degrees of freedom, allowing the scratch model to memorize pixel noise instead of learning generalizable representations).*
3. **Warmup Justification**: Why should the newly initialized head be trained to convergence with a frozen backbone before unfreezing the backbone for fine-tuning?
   *(A randomly initialized head produces large, erratic loss gradients; passing these back into an unfrozen backbone immediately corrupts pre-trained weights).*
4. **Embedding Concept**: How does an industrial "Embedding API" relate to the backbone/head architecture in this lesson?
   *(An embedding API is simply a remote, pre-trained backbone that accepts raw inputs and returns intermediate feature representations, leaving the downstream head to be trained locally).*
5. **Negative Transfer**: Under what circumstances will transfer learning underperform training from scratch?
   *(When the source task distribution is fundamentally mismatched with the target task, causing the backbone to filter out features essential to the target domain).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Frozen backbone updates anyway during training | Passed `model.parameters()` to optimizer without setting `requires_grad = False` | Set `p.requires_grad = False` on backbone parameters and pass `trainable_parameters(model)` |
| `extract_features` returns tensors that consume VRAM / track graphs | Forgot to wrap forward pass in `torch.no_grad()` | Wrap extraction inside `with torch.no_grad():` and call `backbone.eval()` |
| Fine-tuning fails to update backbone parameters | Forgot to restore `requires_grad = True` when unfreezing | Loop over `model.backbone.parameters()` setting `p.requires_grad = True` |
| `few_shot_split` has overlapping train and test examples | Permuted train and test slices independently | Use `np.setdiff1d(np.arange(n), train_indices)` to construct the disjoint test split |
| Test suite skipped during pytest run | PyTorch extra not installed in the active virtual environment | Run `pip install -e ".[torch]"` |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`few_shot_split`**:
  ```python
  train_idx = []
  for c in np.unique(labels):
      c_idx = np.flatnonzero(labels == c)
      train_idx.extend(rng.permutation(c_idx)[:per_class])
  train_idx = np.array(train_idx)
  test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)
  return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
  ```
- **`freeze`**:
  ```python
  for p in module.parameters():
      p.requires_grad = False
  ```
- **`TransferModel.forward`**: `return self.head(self.backbone(X))`
- **`extract_features`**:
  ```python
  backbone.eval()
  with torch.no_grad():
      return backbone(X)
  ```
- **`fine_tune_optimiser`**:
  ```python
  for p in model.parameters():
      p.requires_grad = True
  return torch.optim.Adam([
      {"params": model.head.parameters(), "lr": head_lr},
      {"params": model.backbone.parameters(), "lr": backbone_lr}
  ])
  ```
</details>
