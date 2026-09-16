"""DefEnSyn: neural network defense through synthesis of feature repair programs."""

from __future__ import annotations

import ast
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

__version__ = "0.2.0"
__all__ = [
    "Program",
    "RepairLayer",
    "DefendedModel",
    "SynthesisConfig",
    "SynthesisResult",
    "collect_features",
    "synthesize",
    "load_programs",
]


@dataclass(frozen=True)
class Feature:
    index: int

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Feature indices must be nonnegative integers.")


Operand = Union[Feature, float]

OPERATORS = ("add", "sub", "mul", "div", "min", "max")


@dataclass(frozen=True)
class Instruction:
    target: int
    left: Operand
    operator: str = "assign"
    right: Operand | None = None

    def __post_init__(self):
        Feature(self.target)
        if self.operator not in ("assign", *OPERATORS):
            raise ValueError(f"Unsupported operator: {self.operator}")
        if (self.operator == "assign") != (self.right is None):
            raise ValueError("Assignments take one operand; binary operations take two.")
        for operand in (self.left, self.right):
            if operand is not None and not isinstance(operand, Feature):
                if type(operand) not in (int, float) or not math.isfinite(operand):
                    raise ValueError("Constants must be finite real numbers.")

    @property
    def sources(self) -> frozenset[int]:
        return frozenset(x.index for x in (self.left, self.right) if isinstance(x, Feature))

    def __str__(self):
        def fmt(x):
            return f"x[{x.index}]" if isinstance(x, Feature) else repr(float(x))

        rhs = fmt(self.left)
        if self.operator in ("min", "max"):
            rhs = f"{self.operator}({rhs}, {fmt(self.right)})"
        elif self.operator != "assign":
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.operator]
            rhs = f"{rhs} {symbol} {fmt(self.right)}"
        return f"x[{self.target}] = {rhs}"


def _operand(node: ast.AST) -> Operand:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        index = node.slice
        if node.value.id == "x" and isinstance(index, ast.Constant):
            return Feature(index.value)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if math.isfinite(node.value):
            return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        if isinstance(node.operand, ast.Constant):
            value = _operand(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Operands must be x[index] or a finite numeric constant.")


def parse_instruction(source: str) -> Instruction:
    """Parse only a single assignment in the paper's grammar, with zero-based indices."""
    try:
        body = ast.parse(source.strip()).body
    except (SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid repair instruction: {source!r}") from exc
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
        raise ValueError("Expected one assignment per instruction.")
    assignment = body[0]
    target = _operand(assignment.targets[0])
    if not isinstance(target, Feature):
        raise ValueError("Assignment target must be x[index].")
    rhs = assignment.value
    if isinstance(rhs, ast.BinOp):
        operators = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        operator = operators.get(type(rhs.op))
        if operator is None:
            raise ValueError("Only +, -, *, /, min, and max are supported.")
        return Instruction(target.index, _operand(rhs.left), operator, _operand(rhs.right))
    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
        if rhs.func.id in ("min", "max") and len(rhs.args) == 2 and not rhs.keywords:
            return Instruction(
                target.index, _operand(rhs.args[0]), rhs.func.id, _operand(rhs.args[1])
            )
    return Instruction(target.index, _operand(rhs))


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]

    def __post_init__(self):
        object.__setattr__(self, "instructions", tuple(self.instructions))
        if not self.instructions or not all(isinstance(x, Instruction) for x in self.instructions):
            raise ValueError("A repair program must contain at least one instruction.")

    @classmethod
    def parse(cls, instructions: list[str]) -> Program:
        if not isinstance(instructions, list) or not all(isinstance(x, str) for x in instructions):
            raise ValueError("Program must be a list of instruction strings.")
        return cls(tuple(parse_instruction(x) for x in instructions))

    def to_strings(self) -> list[str]:
        return [str(x) for x in self.instructions]

    @property
    def targets(self) -> frozenset[int]:
        return frozenset(x.target for x in self.instructions)

    @property
    def sources(self) -> frozenset[int]:
        return frozenset().union(*(x.sources for x in self.instructions))

    def validate_dimension(self, dimension: int):
        if max(self.targets | self.sources) >= dimension:
            raise ValueError(f"Program references a feature outside dimension {dimension}.")

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError(
                "Features must be a floating-point tensor with shape [batch, features]."
            )
        self.validate_dimension(features.shape[1])
        output = features.clone()
        for instruction in self.instructions:

            def resolve(operand):
                if isinstance(operand, Feature):
                    return output[:, operand.index]
                return torch.full_like(output[:, instruction.target], float(operand))

            left = resolve(instruction.left)
            op = instruction.operator
            if op == "assign":
                value = left
            else:
                right = resolve(instruction.right)
                if op == "div":
                    # Section 3: skip this instruction for inputs whose divisor is zero.
                    nonzero = right != 0
                    value = torch.where(
                        nonzero,
                        left / torch.where(nonzero, right, torch.ones_like(right)),
                        output[:, instruction.target],
                    )
                else:
                    function = {
                        "add": torch.add,
                        "sub": torch.sub,
                        "mul": torch.mul,
                        "min": torch.minimum,
                        "max": torch.maximum,
                    }[op]
                    value = function(left, right)
            # Functional update preserves sequential semantics and autograd history.
            updated = output.clone()
            updated[:, instruction.target] = value
            output = updated
        return output


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisConfig:
    max_length: int = 10
    samples_per_length: int = 1_000_000
    top_k: int = 30
    clean_weight: float = 1.0
    constant_min: float = -30.0
    constant_max: float = 30.0
    improvement_threshold: float = 1.01
    seed: int = 0
    batch_size: int = 256
    learn_distributions: bool = True

    def __post_init__(self):
        for name in ("max_length", "samples_per_length", "top_k", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if type(self.learn_distributions) is not bool:
            raise ValueError("learn_distributions must be a boolean.")
        for name in ("clean_weight", "constant_min", "constant_max", "improvement_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.clean_weight < 0 or self.constant_min > self.constant_max:
            raise ValueError("clean_weight must be nonnegative and constant_min <= constant_max.")
        if self.improvement_threshold < 1:
            raise ValueError("improvement_threshold must be >= 1 (1 disables early stopping).")


@dataclass(frozen=True)
class ScoredProgram:
    program: Program
    score: float


@dataclass
class SynthesisResult:
    programs: list[ScoredProgram]
    config: SynthesisConfig
    history: list[dict]
    feature_dimension: int
    training_samples: int

    @property
    def repair_programs(self) -> list[Program]:
        """Selected programs in descending score order."""
        return [entry.program for entry in self.programs]

    def save(self, path):
        """Save the selected programs, configuration, and search history as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "feature_dimension": self.feature_dimension,
            "training_samples": self.training_samples,
            "score_definition": "adversarial_correct_count + clean_weight * clean_correct_count",
            "config": asdict(self.config),
            "history": self.history,
            "programs": [
                {"instructions": p.program.to_strings(), "score": p.score} for p in self.programs
            ],
        }


@contextmanager
def evaluation_mode(module: nn.Module):
    """Temporarily evaluate a module without changing the caller's training flags."""
    states = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in states:
            child.training = training


def validate_features(clean: torch.Tensor, adversarial: torch.Tensor, labels: torch.Tensor):
    if clean.ndim != 2 or clean.shape != adversarial.shape or min(clean.shape) < 1:
        raise ValueError(
            "clean and adversarial must have identical nonempty [samples, features] shapes."
        )
    if not clean.is_floating_point() or clean.dtype != adversarial.dtype:
        raise ValueError("Feature tensors must have the same floating-point dtype.")
    if clean.device != adversarial.device or clean.device != labels.device:
        raise ValueError("Features and labels must be on the same device.")
    if labels.shape != (clean.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must have shape [samples] and dtype torch.long.")
    if not torch.isfinite(clean).all() or not torch.isfinite(adversarial).all():
        raise ValueError("Input features must be finite.")
    if (labels < 0).any():
        raise ValueError("Class labels must be nonnegative.")


def _softmax_means(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    average = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    exponentials = np.exp(average - average.max())
    return exponentials / exponentials.sum()


def _sample_instruction(rng, lhs, rhs, config):
    target = int(rng.choice(len(lhs), p=lhs))
    first = Feature(int(rng.choice(len(rhs), p=rhs)))
    second = Feature(int(rng.choice(len(rhs), p=rhs)))
    constant = float(rng.uniform(config.constant_min, config.constant_max))
    operands = (constant, first, second)
    left = operands[int(rng.integers(3))]
    right = operands[int(rng.integers(3))]
    operator = OPERATORS[int(rng.integers(len(OPERATORS)))]
    form = int(rng.integers(3))
    if form == 0:
        return Instruction(target, constant)
    if form == 1:
        return Instruction(target, first)
    return Instruction(target, left, operator, right)


class _TopPrograms:
    """Keep exactly the top K representatives with distinct repaired-feature sets.

    Evicting a representative outside the top K is safe: the cutoff never falls,
    and a later improvement for its feature set can compete as a new candidate.
    This avoids retaining millions of losing programs from Algorithm 1's map.
    """

    def __init__(self, k):
        self.k = k
        self.entries = {}

    def add(self, program, score):
        key = program.targets
        current = self.entries.get(key)
        if current is not None:
            if score > current.score:
                self.entries[key] = ScoredProgram(program, score)
        elif len(self.entries) < self.k:
            self.entries[key] = ScoredProgram(program, score)
        else:
            worst = min(self.entries, key=lambda x: self.entries[x].score)
            if score > self.entries[worst].score:
                del self.entries[worst]
                self.entries[key] = ScoredProgram(program, score)

    def best(self):
        return sorted(self.entries.values(), key=lambda x: x.score, reverse=True)


def _score(program, classifier, clean, adversarial, labels, config):
    score = 0.0
    for inputs, weight in ((clean, config.clean_weight), (adversarial, 1.0)):
        for start in range(0, len(labels), config.batch_size):
            repaired = program(inputs[start : start + config.batch_size])
            if not torch.isfinite(repaired).all():
                return None
            logits = classifier(repaired)
            if logits.ndim != 2 or logits.shape[0] != repaired.shape[0]:
                raise ValueError("Classifier must return [batch, classes] logits.")
            if not torch.isfinite(logits).all():
                return None
            predicted = logits.argmax(dim=1)
            score += weight * (predicted == labels[start : start + config.batch_size]).sum().item()
    return score


def synthesize(
    classifier: nn.Module,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    labels: torch.Tensor,
    config: SynthesisConfig | None = None,
) -> SynthesisResult:
    """Search on cached E(x) and E(A(C(E(x)))); all inputs must share a device.

    The classifier must produce deterministic logits in evaluation mode. Scores
    use counts, as in Algorithm 1, rather than percentages. Overflowing candidates
    are discarded and counted in the history. Input tensors are never modified.
    """
    config = config or SynthesisConfig()
    validate_features(clean, adversarial, labels)
    dimension = clean.shape[1]
    rng = np.random.default_rng(config.seed)
    sums = np.zeros((2, dimension))
    counts = np.zeros((2, dimension), dtype=np.int64)
    distributions = np.full((2, dimension), 1 / dimension)
    top = _TopPrograms(config.top_k)
    history = []
    previous = None

    with evaluation_mode(classifier), torch.no_grad():
        probe = classifier(clean[:1])
        if probe.ndim != 2 or probe.shape[0] != 1 or probe.shape[1] < 1:
            raise ValueError("Classifier must return [batch, classes] logits.")
        if not torch.isfinite(probe).all() or labels.max().item() >= probe.shape[1]:
            raise ValueError("Classifier outputs must be finite and cover all class labels.")
        for length in range(1, config.max_length + 1):
            discarded = 0
            for iteration in range(config.samples_per_length):
                program = Program(
                    tuple(_sample_instruction(rng, *distributions, config) for _ in range(length))
                )
                score = _score(program, classifier, clean, adversarial, labels, config)
                if score is None:
                    discarded += 1
                    continue
                top.add(program, score)
                # Algorithm 1 counts each feature once per program, on each side.
                for side, features in enumerate((program.targets, program.sources)):
                    indices = list(features)
                    sums[side, indices] += score
                    counts[side, indices] += 1
                if (iteration + 1) % 10000 == 0:
                    logger.info(
                        "Length %d: %d/%d candidates",
                        length,
                        iteration + 1,
                        config.samples_per_length,
                    )
            if not top.entries:
                raise ValueError(
                    "All candidates produced nonfinite values; check features and constants."
                )
            if config.learn_distributions:
                distributions = np.stack([_softmax_means(s, c) for s, c in zip(sums, counts)])
            current = sum(p.score for p in top.best())
            # With zero prior score the ratio is undefined: continue exploring.
            stop = (
                previous is not None
                and previous > 0
                and current / previous < config.improvement_threshold
            )
            history.append(
                {
                    "length": length,
                    "sampled": config.samples_per_length,
                    "discarded_nonfinite": discarded,
                    "top_k_score_sum": current,
                    "retained": len(top.entries),
                    "early_stopped": stop,
                }
            )
            logger.info(
                "Length %d complete: best score %.3f; %d programs retained",
                length,
                top.best()[0].score,
                len(top.entries),
            )
            if stop:
                break
            previous = current
    return SynthesisResult(top.best(), config, history, dimension, len(labels))


class RepairLayer(nn.Module):
    """Choose one program independently per input, including in evaluation mode.

    Set a seed to own an independent CPU generator. Its state is included in
    state_dict together with the programs, so a restored layer continues the stream.
    Without a seed, sampling uses PyTorch's global CPU generator.
    """

    def __init__(self, programs: list[Program], seed: int | None = None):
        super().__init__()
        if not programs or not all(isinstance(p, Program) for p in programs):
            raise ValueError("RepairLayer needs at least one Program.")
        self.programs = tuple(programs)
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError(
                "RepairLayer expects floating-point features of shape [batch, features]."
            )
        for program in self.programs:
            program.validate_dimension(features.shape[1])
        if len(self.programs) == 1:
            return self.programs[0](features)
        choices = torch.randint(len(self.programs), (features.shape[0],), generator=self.generator)
        output = features.clone()
        for index in choices.unique().tolist():
            rows = (choices == index).nonzero(as_tuple=True)[0].to(features.device)
            output[rows] = self.programs[index](features[rows])
        return output

    def get_extra_state(self):
        return {
            "programs": [p.to_strings() for p in self.programs],
            "rng_state": None if self.generator is None else self.generator.get_state(),
        }

    def set_extra_state(self, state):
        self.programs = tuple(Program.parse(p) for p in state["programs"])
        if not self.programs:
            raise ValueError("Saved repair layer has no programs.")
        self.generator = None
        if state["rng_state"] is not None:
            self.generator = torch.Generator().set_state(state["rng_state"].cpu())


class DefendedModel(nn.Module):
    """The defended network C(P(E(x))). Call .eval() before evaluation."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        programs: list[Program],
        seed: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.repair = RepairLayer(programs, seed=seed)
        self.classifier = classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.repair(self.encoder(inputs)))


def collect_features(encoder, data_loader, attack, *, max_samples=750, device=None):
    """Encode paired clean/adversarial inputs for synthesis.

    data_loader yields (inputs, labels). attack(inputs, labels) returns perturbed
    inputs for the original network. Include preprocessing in the encoder or
    data pipeline. Returned tensors share the selected device; labels are long.
    """
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None.")
    if device is None:
        sample = next(encoder.parameters(), None)
        if sample is None:
            sample = next(encoder.buffers(), None)
        device = sample.device if sample is not None else "cpu"
    clean_parts, adversarial_parts, label_parts = [], [], []
    collected = 0
    with evaluation_mode(encoder):
        for images, labels in data_loader:
            if max_samples is not None:
                remaining = max_samples - collected
                images, labels = images[:remaining], labels[:remaining]
            images, labels = images.to(device), labels.to(device)
            if len(labels) == 0:
                continue
            with torch.no_grad():
                clean = encoder(images).detach().clone()
            # Attacks may use gradients; keep this call outside torch.no_grad().
            adversarial_images = attack(images.clone(), labels).detach()
            with torch.no_grad():
                adversarial = encoder(adversarial_images).detach().clone()
            validate_features(clean, adversarial, labels)
            clean_parts.append(clean)
            adversarial_parts.append(adversarial)
            label_parts.append(labels.detach().clone())
            collected += len(labels)
            if max_samples is not None and collected >= max_samples:
                break
    if not collected:
        raise ValueError("data_loader did not provide any examples.")
    return torch.cat(clean_parts), torch.cat(adversarial_parts), torch.cat(label_parts)


def load_programs(path) -> list[Program]:
    """Load repair programs from a JSON file written by SynthesisResult.save."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Expected a version 1 DefEnSyn program file.")
    entries = document.get("programs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Program file must contain a nonempty programs list.")
    programs = []
    for entry in entries:
        if not isinstance(entry, dict) or "instructions" not in entry:
            raise ValueError("Each program must contain an instructions list.")
        programs.append(Program.parse(entry["instructions"]))
    return programs
