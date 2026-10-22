# Lesson 7 — Generalisation, initialisation, optimisers

**Read [THEORY.md](THEORY.md) first.**

## What you build

The engineering tools required to train networks effectively rather than just blindly running loops:
- Unbiased `train_val_split` with random permutation.
- Variance-preserving `he_normal` initialisation.
- Three optimisers: `SGD`, `Momentum`, and `Adam`.
- An `EarlyStopping` monitor that saves and restores array clones.
- A comprehensive `fit` training loop recording loss and accuracy trajectories.
- An automated `diagnose` function that classifies training runs as *Diverging*, *Underfitting*, *Overfitting*, or *Healthy*.

The payoff test runs four distinct real training scenarios and validates that `diagnose` labels every trajectory correctly.

## Setup

Run the test suite and demonstration script:

```bash
pytest lesson07_generalisation -x
python lesson07_generalisation/generalisation.py
```

---

## Tasks

| # | What | Target Signature | Theory Reference |
|---|---|---|---|
| 1 | `train_val_split` | `(X, y, val_fraction, rng) -> (X_tr, y_tr, X_val, y_val)` | [THEORY §1](THEORY.md#1-generalisation-the-only-score-that-counts) |
| 2 | `he_normal` | `(d_in, d_out, rng) -> FloatArray` | [THEORY §3](THEORY.md#3-initialisation-scaling-signal-variance) |
| 3 | `SGD.step`, `Momentum.step`, `Adam.step` | Optimiser classes | [THEORY §5](THEORY.md#5-better-ways-to-step-optimizers) |
| 4 | `EarlyStopping.update`, `EarlyStopping.restore` | Validation checkpointing | [THEORY §6](THEORY.md#6-early-stopping) |
| 5 | `fit` | Minibatch loop with evaluation | [THEORY §1, §6](THEORY.md#6-early-stopping) |
| 6 | `diagnose` | `(history) -> str` | [THEORY §7](THEORY.md#7-automated-run-diagnosis) |

### Task 1: `train_val_split`
- Permute indices with `rng.permutation(len(X))`.
- Split into validation slice of size `int(len(X) * val_fraction)` and remaining training slice.
- Guarantees zero sample overlap between training and validation.

### Task 2: `he_normal`
- Draw weights from $\mathcal{N}(0, \sigma^2)$ where $\sigma = \sqrt{\frac{2}{d_{\text{in}}}}$.
- `rng.normal(0.0, math.sqrt(2.0 / d_in), size=(d_in, d_out))`.

### Task 3: Optimizers
- **`SGD.step(model, lr)`**: updates `p -= lr * g`.
- **`Momentum.step(model, lr)`**:
  - `self.velocities` initialized to zeros for each parameter.
  - Updates in-place: `v *= self.beta; v += g; p -= lr * v`.
- **`Adam.step(model, lr)`**:
  - Increment global step: `self.t += 1`.
  - Update first and second moments in-place:
    ```python
    m *= self.beta1
    m += (1.0 - self.beta1) * g
    v *= self.beta2
    v += (1.0 - self.beta2) * (g ** 2)
    ```
  - Compute bias-corrected moments $\hat{m} = \frac{m}{1 - \beta_1^t}$ and $\hat{v} = \frac{v}{1 - \beta_2^t}$.
  - Update: `p -= lr * (m_hat / (np.sqrt(v_hat) + self.eps))`.

### Task 4: `EarlyStopping`
- `update(model, val_loss, epoch)`:
  - If `val_loss < self.best_loss`: update `best_loss`, record `best_epoch = epoch`, and save clones:
    `self.best_params = [p.copy() for p, _ in model.params()]`.
  - Return `True` (stop) if `(epoch - self.best_epoch) >= self.patience`, else `False`.
- `restore(model)`:
  - Restore weights in place:
    `for (p, _), best in zip(model.params(), self.best_params): p[...] = best`.

### Task 5: `fit`
- Minibatch training loop tracking epoch metrics:
  - Run batches via `minibatches`.
  - Evaluate whole-set loss and accuracy for both train and validation sets at epoch end.
  - Feed `val_loss` into `early_stopping.update()`.
  - If early stopping triggers: break and call `early_stopping.restore()`.

### Task 6: `diagnose`
- Inspect `history` strictly in prioritized order:
  1. `Diverging`: final loss is `nan`, `inf`, or `history.train_loss[-1] > history.train_loss[0]`.
  2. `Underfitting`: final training accuracy `history.train_acc[-1] < 0.90`.
  3. `Overfitting`: `(history.train_acc[-1] - history.val_acc[-1]) >= 0.10`.
  4. Otherwise return `Healthy`.

---

## Experiments

1. **Initialisation Breakdown**:
   Construct a 10-hidden-layer network ($d_{\text{in}} = d_{\text{out}} = 100$) using `uniform(-1, 1)` initialisation. Attempt to train on digits. What happens to the loss?
2. **Learning Rate Sensitivity**:
   Compare SGD over $lr \in \{0.001, 0.01, 0.1, 1.0, 5.0\}$ against Adam over $lr \in \{0.0001, 0.001, 0.01, 0.1\}$. Which optimizer converges across a broader range of learning rates?
3. **Data Volume vs Model Capacity**:
   In the overfitting demo, compare training a 256-unit MLP on 60 images vs 600 images vs training a 16-unit MLP on 60 images. Which intervention shrinks the generalization gap most effectively?
4. **Weight Decay Regularization**:
   Add an $L_2$ weight penalty to parameter gradients before calling the optimizer:
   `g += 2.0 * 1e-3 * p`. Observe the impact on validation accuracy on the 60-image subset.

---

## Self-Check Questions

1. **Test Set Integrity**: If you iterate on model architecture and learning rate based on validation accuracy 50 times, is your final validation score still an unbiased estimate of real-world generalization?
   *(No. Repeatedly tuning hyperparameters to maximize validation accuracy causes meta-overfitting to the validation set. An untouched test set is required for the final honest evaluation).*
2. **Group Leakage**: A medical imaging dataset has 10 chest X-rays for each of 200 patients. Why is a standard random row split invalid?
   *(Photos from the same patient share identical anatomy, sensor noise, and idiosyncratic markers. If a patient's images appear in both train and validation sets, the model learns patient identification rather than pathology).*
3. **Scale Invariance in Adam**: Why does dividing by $\sqrt{\hat{v}_t} + \epsilon$ make Adam's effective step size largely independent of the raw gradient magnitude?
   *($\hat{m}_t$ scales linearly with $g$, while $\sqrt{\hat{v}_t}$ also scales linearly with $|g|$. Their ratio $\frac{\hat{m}}{\sqrt{\hat{v}}}$ scales as a dimensionless unit $\approx \pm 1$, decoupling step size from gradient scale).*
4. **Cloning Requirement**: In early stopping, why must you call `p.copy()` instead of storing `p`?
   *(`p` is a reference to a mutable NumPy array; storing references means `self.best_params` continually updates as training progresses, destroying the saved checkpoint).*
5. **Diagnostic Ordering**: Why does `diagnose` evaluate *Diverging* before *Underfitting*?
   *(Because a diverged run with `loss = nan` will also fail the accuracy threshold $< 0.90$. If tested in reverse, a diverging run would be falsely misdiagnosed as underfitting).*

---

## Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Momentum or Adam velocities never update | Reassigned array (`v = beta * v + g`) breaking list references | Mutate in place: `v *= beta; v += g` |
| Adam's early steps are minuscule | Omitted bias correction factors $(1 - \beta^t)$ | Compute $\hat{m} = \frac{m}{1 - \beta_1^t}$ and $\hat{v} = \frac{v}{1 - \beta_2^t}$ |
| Adam step counter advances too fast | Incremented `self.t` inside the parameter loop | Increment `self.t += 1` once per `step()` call |
| `EarlyStopping` fails to restore best weights | Stored references or rebound variable (`p = best`) | Use `p.copy()` when saving, and `p[...] = best` when restoring |
| `train_val_split` test fails with overlap | Sliced overlapping index ranges or didn't permute | Use disjoint slices: `order[:n_val]` and `order[n_val:]` |
| `diagnose` classifies a `NaN` run as underfitting | Evaluated accuracy before checking for divergence | Check `np.isnan(loss)` and divergence first in the decision tree |

---

<details>
<summary>Hints (open only after trying)</summary>

- **`train_val_split`**:
  ```python
  n_val = int(len(X) * val_fraction)
  order = rng.permutation(len(X))
  val_idx, train_idx = order[:n_val], order[n_val:]
  return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
  ```
- **`he_normal`**: `rng.normal(0.0, math.sqrt(2.0 / d_in), size=(d_in, d_out))`.
- **`Adam.step`**: Increment `self.t += 1` first. Loop over `zip(model.params(), self.m, self.v)`.
- **`EarlyStopping`**: Save `[p.copy() for p, _ in model.params()]`. Restore with `p[...] = best`.
- **`diagnose`**:
  ```python
  if math.isnan(history.train_loss[-1]) or math.isinf(history.train_loss[-1]) or history.train_loss[-1] > history.train_loss[0]:
      return "diverging"
  if history.train_acc[-1] < 0.90:
      return "underfitting"
  if history.train_acc[-1] - history.val_acc[-1] >= 0.10:
      return "overfitting"
  return "healthy"
  ```
</details>
