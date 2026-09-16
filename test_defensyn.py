import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from defensyn import (
    DefendedModel,
    Program,
    RepairLayer,
    SynthesisConfig,
    _softmax_means,
    _TopPrograms,
    collect_features,
    load_programs,
    synthesize,
)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("x[1]", [2.0, 0.0]),
        ("-1.5", [-1.5, -1.5]),
        ("x[0] + x[1]", [6.0, 3.0]),
        ("x[0] - x[1]", [2.0, 3.0]),
        ("x[0] * x[1]", [8.0, 0.0]),
        ("x[0] / x[1]", [2.0, 3.0]),
        ("min(x[0], x[1])", [2.0, 0.0]),
        ("max(x[0], x[1])", [4.0, 3.0]),
        ("-2 - x[1]", [-4.0, -2.0]),
        ("x[0] - -2", [6.0, 5.0]),
        ("-2 / x[1]", [-1.0, 3.0]),
        ("max(-2, 1e-3)", [0.001, 0.001]),
    ],
)
def test_language_semantics(expression, expected):
    features = torch.tensor([[4.0, 2.0], [3.0, 0.0]])
    before = features.clone()
    output = Program.parse([f"x[0] = {expression}"])(features)
    torch.testing.assert_close(output[:, 0], torch.tensor(expected))
    torch.testing.assert_close(features, before)


def test_sequential_execution_and_autograd():
    program = Program.parse(["x[0] = x[1] * 2", "x[1] = x[0] + 1"])
    features = torch.tensor([[10.0, 3.0]], requires_grad=True)
    output = program(features)
    torch.testing.assert_close(output, torch.tensor([[6.0, 7.0]]))
    output.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[0.0, 4.0]]))


def test_zero_division_skips_only_affected_instruction_and_row():
    program = Program.parse(["x[0] = 5", "x[0] = 2 / x[1]", "x[1] = x[0] + 1"])
    result = program(torch.tensor([[9.0, 0.0], [9.0, 2.0]]))
    torch.testing.assert_close(result, torch.tensor([[5.0, 6.0], [1.0, 2.0]]))


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "x[0] = __import__('os').system('echo no')",
        "x[0] = x[-1]",
        "x[0] = x[True]",
        "x[0] = 1e999",
        "x[0] = x[1] ** 2",
        "x[0] = x[1].item()",
        "x[0] = x[1] + x[2] + 1",
        "x[0] = 1; x[1] = 2",
        "x[0] = max(x[1], key=1)",
    ],
)
def test_rejects_expressions_outside_the_dsl(source):
    with pytest.raises(ValueError):
        Program.parse([source])


def test_dimension_and_program_roundtrip():
    program = Program.parse(["x[3] = min(x[0], -2.5)"])
    assert Program.parse(program.to_strings()) == program
    with pytest.raises(ValueError, match="dimension"):
        program(torch.zeros(2, 3))


def test_sampling_is_per_input_and_continues_in_eval_mode():
    programs = [Program.parse(["x[0] = 1"]), Program.parse(["x[0] = 2"])]
    layer = RepairLayer(programs, seed=42).eval()
    batch = torch.zeros(128, 1)
    first = layer(batch)
    assert set(first.flatten().tolist()) == {1.0, 2.0}
    assert not torch.equal(first, layer(batch))
    torch.testing.assert_close(first, RepairLayer(programs, seed=42)(batch))


def test_repair_state_dict_preserves_programs_and_rng_position():
    programs = [Program.parse(["x[0] = 1"]), Program.parse(["x[0] = 2"])]
    original = RepairLayer(programs, seed=42)
    original(torch.zeros(16, 1))
    state = original.state_dict()
    restored = RepairLayer([Program.parse(["x[0] = 0"])])
    restored.load_state_dict(state)
    torch.testing.assert_close(original(torch.zeros(16, 1)), restored(torch.zeros(16, 1)))


def fixture_data():
    features = torch.tensor([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
    adversarial = features.clone()
    adversarial[:, 0] *= -1
    labels = torch.tensor([0, 0, 1, 1])
    classifier = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    return classifier, features, adversarial, labels


def test_synthesis_finds_repair_and_is_reproducible():
    head, clean, adversarial, labels = fixture_data()
    config = SynthesisConfig(max_length=2, samples_per_length=120, top_k=2, seed=7)
    original = clean.clone()
    state = head.weight.detach().clone()
    first = synthesize(head, clean, adversarial, labels, config)
    second = synthesize(head, clean, adversarial, labels, config)
    assert first.to_dict() == second.to_dict()
    assert first.programs[0].score == 8
    assert len({p.program.targets for p in first.programs}) == len(first.programs)
    best = first.programs[0].program
    assert torch.equal(head(best(adversarial)).argmax(1), labels)
    assert head.training  # The caller's mode is restored.
    torch.testing.assert_close(clean, original)
    torch.testing.assert_close(head.weight, state)


def test_top_k_matches_full_dictionary_even_after_eviction_and_improvement():
    top = _TopPrograms(3)
    reference = {}
    rng = np.random.default_rng(7)
    for iteration in range(500):
        lhs = int(rng.integers(10))
        program = Program.parse([f"x[{lhs}] = {iteration}"])
        score = float(rng.random())
        reference[lhs] = max(reference.get(lhs, -1), score)
        top.add(program, score)
        assert [p.score for p in top.best()] == sorted(reference.values(), reverse=True)[:3]


def test_softmax_is_finite_for_large_scores_and_unseen_features():
    result = _softmax_means(np.array([1500.0, 1500.0, 0.0]), np.array([1, 1, 0]))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [0.5, 0.5, 0.0])


def test_early_stopping_when_best_score_cannot_improve():
    head = nn.Linear(1, 1)
    clean = torch.ones(3, 1)
    labels = torch.zeros(3, dtype=torch.long)
    result = synthesize(
        head, clean, clean, labels, SynthesisConfig(max_length=5, samples_per_length=8, top_k=1)
    )
    assert len(result.history) == 2
    assert result.history[-1]["early_stopped"]


def test_zero_scores_do_not_divide_by_zero_or_stop_early():
    head = nn.Linear(1, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 1.0]))
    clean = torch.ones(3, 1)
    labels = torch.zeros(3, dtype=torch.long)
    result = synthesize(
        head, clean, clean, labels, SynthesisConfig(max_length=3, samples_per_length=5, top_k=1)
    )
    assert len(result.history) == 3
    assert all(p.score == 0 for p in result.programs)


@pytest.mark.parametrize(
    "values",
    [
        {"samples_per_length": 0},
        {"top_k": -1},
        {"max_length": True},
        {"constant_min": 3, "constant_max": 2},
        {"clean_weight": -1},
        {"clean_weight": float("nan")},
        {"seed": -1},
        {"learn_distributions": "yes"},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        SynthesisConfig(**values)


def test_invalid_inputs_and_classifier_modes_are_preserved_on_error():
    head, clean, adversarial, labels = fixture_data()
    with pytest.raises(ValueError, match="labels"):
        synthesize(head, clean, adversarial, labels.float())
    with pytest.raises(ValueError, match="class labels"):
        synthesize(head, clean, adversarial, labels + 10)
    assert head.training


def test_feature_collection_pairs_inputs_and_honors_sample_limit():
    encoder = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(2))
    inputs = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    labels = torch.zeros(7, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)
    seen = []

    def attack(images, labels):
        assert torch.is_grad_enabled()
        seen.append(len(images))
        return images.add_(2)

    clean, adversarial, actual_labels = collect_features(encoder, loader, attack, max_samples=5)
    torch.testing.assert_close(clean, inputs[:5])
    torch.testing.assert_close(adversarial, inputs[:5] + 2)
    assert seen == [3, 2]
    assert actual_labels.shape == (5,)
    assert encoder.training
    assert not clean.requires_grad


def test_feature_collection_rejects_empty_data_and_restores_mode():
    encoder = nn.Identity()
    with pytest.raises(ValueError, match="did not provide"):
        collect_features(encoder, [], lambda x, y: x)
    assert encoder.training
    with pytest.raises(ValueError, match="max_samples"):
        collect_features(encoder, [], lambda x, y: x, max_samples=0)


def test_save_load_and_model_composition(tmp_path):
    head, clean, adversarial, labels = fixture_data()
    result = synthesize(
        head,
        clean,
        adversarial,
        labels,
        SynthesisConfig(max_length=1, samples_per_length=120, top_k=1, seed=7),
    )
    path = tmp_path / "repair/programs.json"
    result.save(path)
    programs = load_programs(path)
    assert programs == result.repair_programs
    assert json.loads(path.read_text())["config"]["top_k"] == 1
    model = DefendedModel(nn.Identity(), head, programs).eval()
    torch.testing.assert_close(model(adversarial).argmax(1), labels)


def test_example_checkpoint_roundtrip(tmp_path):
    from example import run

    first = run(tmp_path)
    second = run(tmp_path / "loaded", checkpoint=tmp_path / "checkpoint.pt")
    assert first == second
    assert first["Defended"]["clean"] == 100
    assert first["Defended"]["perturbed"] == 100
