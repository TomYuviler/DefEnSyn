"""Run a small feature-repair example: python example.py --output runs/example."""

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from defensyn import DefendedModel, Program, SynthesisConfig, collect_features, synthesize


def make_data(samples, generator):
    labels = torch.randint(2, (samples,), generator=generator)
    signal = (2 * labels - 1) * (1 + torch.rand(samples, generator=generator))
    inputs = torch.stack([signal, signal, torch.randn(samples, generator=generator)], dim=1)
    return TensorDataset(inputs, labels)


def perturb(inputs, labels):
    result = inputs.clone()
    result[:, 0] *= -1
    return result


def run(output, seed=0, checkpoint=None):
    generator = torch.Generator().manual_seed(seed)
    encoder = nn.Identity()
    classifier = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    original = nn.Sequential(encoder, classifier).eval()
    train = DataLoader(make_data(96, generator), batch_size=32)
    clean, adversarial, labels = collect_features(encoder, train, perturb)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        classifier.load_state_dict(saved["classifier"])
        programs = [Program.parse(p) for p in saved["programs"]]
    else:
        result = synthesize(
            classifier,
            clean,
            adversarial,
            labels,
            SynthesisConfig(
                max_length=2,
                samples_per_length=200,
                top_k=1,
                constant_min=-3,
                constant_max=3,
                seed=seed,
            ),
        )
        result.save(Path(output) / "programs.json")
        programs = result.repair_programs
        torch.save(
            {
                "classifier": classifier.state_dict(),
                "programs": [p.to_strings() for p in programs],
                "seed": seed,
            },
            Path(output) / "checkpoint.pt",
        )
        print(f"Saved programs and checkpoint to {output}\n")
    defended = DefendedModel(encoder, classifier, programs, seed=seed).eval()

    # Evaluate on a separate synthetic test set.
    test_inputs, test_labels = make_data(256, generator).tensors
    test_adversarial = perturb(test_inputs, test_labels)
    metrics = {}
    with torch.no_grad():
        for name, model in (("Original", original), ("Defended", defended)):
            clean_acc = (model(test_inputs).argmax(1) == test_labels).float().mean().item() * 100
            perturbed_acc = (
                model(test_adversarial).argmax(1) == test_labels
            ).float().mean().item() * 100
            metrics[name] = {"clean": clean_acc, "perturbed": perturbed_acc}
            print(
                f"{name:8s} | clean: {clean_acc:6.2f}% | fixed perturbation: {perturbed_acc:6.2f}%"
            )
    print("\nSelected repair:")
    print("\n".join(programs[0].to_strings()))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/example")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, help="Load a saved example checkpoint")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(arguments.output, arguments.seed, arguments.checkpoint)
