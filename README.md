# DefEnSyn

**Neural network robustness through synthesis of repair programs.**

Code for [Enhancing Neural Network Robustness via Synthesis of Repair Programs](https://doi.org/10.1007/978-3-032-07106-4_10), by **Tom Yuviler** and **Dana Drachsler-Cohen** · **SAS 2025**.

DefEnSyn inserts a short repair program between a trained network's encoder and classifier. It learns which features to repair and retains a diverse set of programs, selecting one independently for each input during inference.

```text
input  →  encoder  →  repair program  →  classifier  →  prediction
```

## Quick start

Python 3.9 or later:

```bash
python -m pip install .
python example.py --checkpoint example_checkpoint.pt
```

The included checkpoint contains a small synthetic classifier and its repair program. To synthesize a new repair and save a checkpoint:

```bash
python example.py --output runs/example
```

## Plug in your network

Load your trained model and split it into two PyTorch modules:

- **`encoder`** maps inputs to feature vectors of shape `[batch, features]`.
- **`classifier`** maps those vectors to logits of shape `[batch, classes]`.

The composition `classifier(encoder(inputs))` should give the original model's output. Use the model's input preprocessing and matching dataset class order.

```python
import torch
from defensyn import DefendedModel, SynthesisConfig, collect_features, synthesize

# Supply your encoder, classifier, training_loader, and attack.
# training_loader yields (inputs, labels), with torch.long labels.
# attack(inputs, labels) returns adversarial inputs for the original model.
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = encoder.to(device).eval()
classifier = classifier.to(device).eval()

clean, adversarial, labels = collect_features(
    encoder,
    training_loader,
    attack,
    max_samples=750,
    device=device,
)
result = synthesize(classifier, clean, adversarial, labels, SynthesisConfig())
result.save("runs/repair/programs.json")

defended = DefendedModel(
    encoder,
    classifier,
    result.repair_programs,
    seed=0,
).eval()
predictions = defended(inputs.to(device)).argmax(dim=1)
```

For example, download and split a torchvision ResNet-18:

```python
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
preprocess = weights.transforms()
classifier = model.fc
model.fc = nn.Identity()
encoder = model
```

Install `torchvision` separately for that example. Other architectures use the same encoder/classifier interface. For local model weights, instantiate the architecture and load its state dictionary with `model.load_state_dict(torch.load(path, weights_only=True))` before splitting it.

Use your chosen attack library with the original model for synthesis and with `defended` for evaluation. When your data pipeline normalizes images, configure the attack with the same normalization so its perturbation bounds apply in pixel space. Evaluate on a separate test set, keeping the repair layer active during attack queries.

## Configuration

Set options directly in `SynthesisConfig`:

| Option | Default |
| :--- | ---: |
| `max_length` | 10 |
| `samples_per_length` | 1,000,000 |
| `top_k` | 30 |
| `clean_weight` | 1.0 |
| `constant_min`, `constant_max` | −30, 30 |
| `improvement_threshold` | 1.01 |
| `seed` | 0 |
| `batch_size` | 256 |

For a small initial run, use `SynthesisConfig(samples_per_length=200, max_length=2)`. Set `learn_distributions=False` for uniform feature sampling.

## Save and load repairs

```python
from defensyn import DefendedModel, load_programs

programs = load_programs("runs/repair/programs.json")
defended = DefendedModel(encoder, classifier, programs, seed=0).eval()
```

Save model weights separately using PyTorch's `state_dict()`. Repair layers also store their programs and random-generator state in their state dictionary.

## Files

```text
defensyn.py             Repair language, synthesis, and model integration
example.py              Runnable synthetic example
example_checkpoint.pt   Small example classifier and repair program
test_defensyn.py        Tests
pyproject.toml          Installation and dependencies
README.md              Setup and usage
CITATION.cff            Paper citation
.gitignore             Excludes generated data and model weights
```

Run tests with `python -m pip install ".[test]"` followed by `python -m pytest`.

## Citation

```bibtex
@inproceedings{yuviler2026defensyn,
  title     = {Enhancing Neural Network Robustness via Synthesis of Repair Programs},
  author    = {Yuviler, Tom and Drachsler-Cohen, Dana},
  booktitle = {SAS 2025},
  year      = {2021},
  pages     = {221--248},
  doi       = {10.1007/978-3-032-07106-4_10}
}
```
