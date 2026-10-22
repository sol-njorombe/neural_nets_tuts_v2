# Lesson 10 — Theory: transfer learning

Lesson 9 ended with a problem: training from scratch needs a lot of labelled
data, and most real projects have little. This lesson shows the standard answer,
and the way most practical neural-network projects start today: take a model
someone already trained on a large dataset, keep what it learned, and train only
a small new part on your own data.

## 1. Concept: reuse what was already learned

**Analogy.** An experienced portrait photographer takes up wildlife photography.
They do not relearn exposure, focus and composition from zero; those skills
carry over. They only need to learn what is new: animal behaviour, long lenses,
patience. With that head start they get good far faster than a beginner would.

A trained network is the same. Its early layers turn raw input into general
**features**; its last layer maps those features to the answers of one specific
task. The features are usually useful far beyond that task.

```
         ┌──────────── backbone ────────────┐   ┌─ head ─┐
pixels ─▶│ Linear → ReLU → Linear → ReLU    │──▶│ Linear │──▶ answers
         └──────────────────────────────────┘   └────────┘
          general: "what is in this image?"      specific: "which of these
          trained once, on lots of data          classes?" trained on yours
```

## 2. Why it works

Lesson 5 showed a hidden layer **re-describing** the inputs so that the output
neuron's job becomes easy. A backbone trained on a large, varied task has
learned a re-description that makes many related tasks easy. For images, the
early layers of large models learn edges, textures and shapes; later ones learn
parts and objects. A new task built on those needs only a simple head and a few
examples.

The same idea runs through modern AI:

- **Image models** (ResNet, ViT, DINOv2, CLIP) are pretrained on millions of
  images and reused as feature extractors.
- **Embeddings** are exactly these features. An "embedding API" is a backbone
  someone else runs for you.
- **Fine-tuning an LLM** is transfer learning on a very large backbone.

## 3. The experiment in this lesson

To see the effect with a small, fast dataset:

| | Source task (pretraining) | Target task (yours) |
|---|---|---|
| Question | Which digit is this? (10 classes) | Is the digit even or odd? (2 classes) |
| Images | The 1,437 training images | The 360 validation images: **different images** |
| Labels | All 1,437 | Only `per_class` of each class, e.g. 25 even + 25 odd |

Keeping the images separate matters: otherwise the backbone would have seen the
test images during pretraining, and the comparison would be leakage (lesson 7 §1).

Three models, the same architecture, trained on the same few labels:

| Model | Starting point | What trains |
|---|---|---|
| From scratch | random backbone | everything |
| Frozen backbone | pretrained backbone | only the new head |
| Fine-tuned | the frozen model above | everything, backbone with a smaller learning rate |

Measured over five seeds (test accuracy on the remaining images):

| Labels per class | From scratch | Frozen backbone | Then fine-tuned |
|---|---|---|---|
| 5 | 0.74 | 0.80 | 0.81 |
| 10 | 0.80 | 0.85 | 0.86 |
| 25 | 0.85 | 0.90 | 0.92 |

The pretrained features help at every size. With 5 labels per class, the frozen
model matches the scratch model trained on twice as many.

The effect here is modest because 8×8 digits are a very small problem: even
raw pixels are fairly good features. With real photos and a backbone trained on
millions of images, the gap is usually dramatic.

## 4. Feature extraction: freeze the backbone

```python
freeze(backbone)                                   # p.requires_grad = False for every parameter
model = TransferModel(backbone, n_features=32, n_classes=2)
optimiser = Adam(trainable_parameters(model))      # only the head's weight and bias
```

`requires_grad = False` tells autograd to stop at those tensors: no `.grad`, no
updates. Only the head's 66 parameters learn, from 50 examples, which is a much
more reasonable ratio than 6,300 parameters from 50 examples.

Because a frozen backbone never changes, its output for each image never changes
either. So for a large backbone you can run it **once** over your data
(`extract_features`), save the features, and train the head on them directly,
which turns hours of training into seconds.

| Math | Shape | Code |
|---|---|---|
| features = backbone(X) | (n, 32) | `extract_features(backbone, X)` |
| logits = features @ Wᵀ + b | (n, 2) | `model.head(features)` |

## 5. Fine-tuning: then adjust the backbone gently

Once the head is sensible, unfreeze the backbone and keep training everything,
with the backbone's learning rate around 10× smaller than the head's. PyTorch
optimisers accept **parameter groups** with different learning rates:

```python
Adam([{"params": model.head.parameters(),     "lr": 0.005},
      {"params": model.backbone.parameters(), "lr": 0.0005}])
```

Why train the head first? A random head sends large, random gradients back into
the backbone and can wreck its features in the first few steps. Why a small
backbone rate? To adjust what it learned, not overwrite it with 50 examples.

## 6. When transfer fails

Transfer only helps if the source task's features are relevant to the target.
Experiment 3 pretrains on digits 0–4 only, then asks for digits 5–9 with 5
labels each. When this lesson was written, five seeds gave average accuracies of
0.89 from scratch and 0.57 with the frozen backbone.
The backbone learned to tell 0–4 apart, and threw away whatever was needed for
5–9. A backbone trained on a broad task (all ten digits, or millions of varied
photos) generalises; one trained on a narrow task may not.

## 7. The real-world version

With a large pretrained image model from `torchvision` (`pip install torchvision`;
the weights download on first use):

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
backbone = resnet50(weights=weights)
backbone.fc = torch.nn.Identity()          # remove its 1000-class head: it now outputs 2048 features
backbone.eval()
preprocess = weights.transforms()          # resizing and normalisation it was trained with

with torch.no_grad():
    features = backbone(preprocess(image).unsqueeze(0))     # (1, 2048), for one PIL image

head = torch.nn.Linear(2048, n_outputs)    # your task: classes, or numbers
```

The rest is this lesson: extract features once, train the head, then fine-tune
if you have enough data.

**A regression head.** Not every task is classification. To predict numbers
(say, the Lightroom slider values a photographer chose for each photo) the head
outputs those numbers directly and the loss is mean squared error, as in lessons
2–5. Scale each target to a similar range first (for example, subtract its mean
and divide by its standard deviation), or the sliders with large numeric ranges
dominate the loss. Lesson 11's first capstone brief works through such a project.
