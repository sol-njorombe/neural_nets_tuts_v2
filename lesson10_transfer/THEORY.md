# Lesson 10 — Theory: transfer learning

Lesson 9 ended with a practical reality: training deep architectures from scratch requires massive volumes of labelled
data, whereas most real-world engineering projects operate under severe data constraints. This lesson presents the standard industry paradigm: take a model
already pretrained on a large, diverse dataset, freeze and preserve its representations, and train only
a specialized task head on your target data.

---

## 0. Learning Objectives

By the end of this lesson, you will be able to:

1. **Deconstruct the Backbone/Head Architecture**: Explain how pre-trained neural networks partition into a general feature extractor (backbone) and task-specific classification/regression layer (head).
2. **Implement Feature Extraction with Frozen Weights**: Implement parameter freezing via `requires_grad = False` and explain why pre-computing representations (`extract_features`) provides $100\times$ speedups.
3. **Configure Multi-Rate Fine-Tuning**: Implement differential learning rates using PyTorch parameter groups, justifying why backbone rates are set $10\times$ smaller than head rates.
4. **Prevent Leakage in Transfer Benchmarking**: Construct disjoint source and target task datasets ensuring zero image overlap between pretraining and few-shot evaluation.
5. **Diagnose Negative Transfer**: Identify when pre-trained feature representations fail to generalize to target domains due to severe feature distribution shift.

---

## 1. Concept: reuse what was already learned

### The Photographer Analogy
An experienced portrait photographer takes up wildlife photography.
They do not relearn exposure, shutter speed, optical depth-of-field, and lighting from scratch; those foundational visual principles transfer directly.
They only need to master the delta: animal tracking, telephoto stabilization, and patience.
With that head start, they achieve mastery in days rather than years.

A deep neural network functions identically:
- **Early layers**: learn universal low-level visual primitives (edges, gradients, color contrasts, basic textures).
- **Intermediate layers**: combine edges into motifs, contours, and surface patterns.
- **Deep layers (Head)**: map task-agnostic representations to specific target labels.

```
                  ┌──────────── backbone ────────────┐   ┌─ head ─┐
pixels (X) ──────▶│ Linear → ReLU → Linear → ReLU    │──▶│ Linear │──▶ Predictions
                  └──────────────────────────────────┘   └────────┘
                    General representation extractor        Task-specific
                    Trained once on massive source data     Trained on small target data
```

---

## 2. Representation Geometry & Modern AI

Lesson 5 proved that a hidden layer operates by **re-describing** the inputs so that the output
decision boundary becomes linearly separable.
A backbone pre-trained on a diverse source task discovers an embedding geometry that makes thousands of downstream tasks linearly separable.

This principle underpins the entire modern foundation model paradigm:
- **Computer Vision**: Backbones (ResNet, ConvNeXt, Vision Transformers, CLIP, DINOv2) pre-trained on ImageNet or web scrape images are reused as frozen feature extractors.
- **Embeddings**: An "Embedding API" (e.g. OpenAI or Gemini text embeddings) is simply a massive frozen backbone executed on cloud infrastructure.
- **LLM Fine-Tuning**: LoRA (Low-Rank Adaptation) and parameter-efficient fine-tuning are transfer learning applied to frozen transformer backbones.

---

## 3. The Controlled Transfer Experiment

To demonstrate transfer learning quantitatively under clean experimental conditions:

| Parameter | Source Pretraining Task | Target Few-Shot Task |
|---|---|---|
| **Objective** | Classify handwritten digits (10 classes: 0–9) | Classify digit parity (2 classes: Even vs Odd) |
| **Image Pool** | 1,437 training images | 360 validation images (**strictly disjoint images**) |
| **Label Budget** | All 1,437 labelled examples | Only `per_class` labels (e.g. 5, 10, or 25 examples per class) |

Partitioning the image pools guarantees zero data leakage: the backbone never sees target images during pretraining.

### Empirical Benchmarks Across 5 Random Seeds

| Label Budget per Class | From Scratch (Random Init) | Frozen Backbone (Transfer) | Fine-Tuned (Differential $lr$) |
|---|---|---|---|
| **5 examples** | $0.74 \pm 0.04$ | **$0.80 \pm 0.02$** | **$0.81 \pm 0.02$** |
| **10 examples** | $0.80 \pm 0.03$ | **$0.85 \pm 0.02$** | **$0.86 \pm 0.02$** |
| **25 examples** | $0.85 \pm 0.02$ | **$0.90 \pm 0.01$** | **$0.92 \pm 0.01$** |

With just 5 labels per class, the frozen backbone achieves performance that requires more than double the data budget when training from scratch!

---

## 4. Feature Extraction: Freezing the Backbone

```python
def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
```

Setting `p.requires_grad = False` instructs the autograd engine:
1. Do not allocate intermediate gradient memory buffers for backbone parameters.
2. Terminate backward graph tracking at the backbone boundary.
3. Freeze weights in place: only the newly initialized head parameters update.

### Parameter Efficiency
In our architecture:
- Total network parameters: $\approx 6,300$
- Head parameters (`Linear(32, 2)`): $32 \times 2 + 2 = \mathbf{66}$ parameters.

Fitting 66 parameters on 50 training images has an optimal ratio of $\approx 1.3$ parameters per example, virtually eliminating the severe overfitting that occurs when fitting 6,300 parameters on 50 examples.

### Offline Feature Caching
Because the frozen backbone is deterministic, you can execute it **once** across your target dataset in evaluation mode:

```python
with torch.no_grad():
    features = backbone(X_train)  # shape: (n, d_features)
```

Training the downstream head directly on cached features reduces multi-epoch training time from minutes to milliseconds.

---

## 5. Fine-Tuning: Differential Learning Rates

Once the newly initialized head has converged to reasonable weights, you can optionally unfreeze the backbone and train the entire network end-to-end.

```python
optimiser = torch.optim.Adam([
    {"params": model.head.parameters(),     "lr": 0.005},
    {"params": model.backbone.parameters(), "lr": 0.0005}  # 10x smaller
])
```

### Why Must the Head Be Trained Before Unfreezing the Backbone?
At initialization, the head weights are random. If backpropagated immediately into an unfrozen backbone, large random gradients wreck the delicate pre-trained feature detectors in the first few gradient steps (**catastrophic forgetting**).

### Why Must the Backbone Learning Rate Be $10\times$ Smaller?
The pre-trained backbone already possesses near-optimal representations. The objective of fine-tuning is gentle alignment with target domain idiosyncrasies, not overwriting pre-existing visual representations with small sample batches.

---

## 6. When Transfer Fails: Negative Transfer

Transfer learning is not guaranteed to improve performance. It fails when the source representation space does not span the target problem:

If you pretrain a backbone strictly on digits $0$ through $4$, and then attempt few-shot transfer to classify digits $5$ through $9$:
- Scratch model accuracy: $\mathbf{0.89}$
- Frozen backbone accuracy: $\mathbf{0.57}$ (little better than random chance!)

The backbone learned features specialized in distinguishing rounded loops and single vertical strokes ($0, 1, 2, 3, 4$), discarding topological features necessary to separate $5, 6, 7, 8, 9$. When source features are orthogonal or irrelevant to the target task, freezing the backbone results in **negative transfer**.

---

## 7. Production torchvision Workflow

In production computer vision, you reuse industrial backbones (e.g. ResNet-50 trained on 1.28 million ImageNet photos):

```python
import torch
import torchvision.models as models

# 1. Load pre-trained architecture with modern weights
weights = models.ResNet50_Weights.DEFAULT
backbone = models.resnet50(weights=weights)

# 2. Strip original 1,000-class ImageNet classification head
backbone.fc = torch.nn.Identity()  # Outputs 2,048-dim feature vector
backbone.eval()

# 3. Attach custom application head
head = torch.nn.Linear(2048, num_classes)
```
