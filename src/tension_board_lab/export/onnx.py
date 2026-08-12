"""Export the grade critic and the generator to ONNX for onnxruntime-web.

Two things here fail quietly rather than loudly:

**Both batch and the variable length need dynamic axes.** The exporter otherwise bakes in the
shapes it happened to trace, and the model then silently works only for problems of exactly
that size. The parity test checks 2, 10, and 35 holds for this reason.

**The critic's mask input is bool.** Passing bools across the JS boundary is awkward, so the
export wrapper takes a float mask and thresholds it inside the graph.

On fp16: the plan warned that masking attention with ``torch.finfo(dtype).min`` would overflow
in half precision and return NaN. Measured, it does not—PyTorch's softmax subtracts the row
maximum and saturates, so both models run in fp16 with logits within 0.01 of fp32 and identical
argmaxes. ``--precision`` still offers only fp32 and int8, because int8 is a quarter of the
download and there is no reason to prefer fp16 over it. If that ever changes, note that
onnxruntime-web's fp16 kernels are a different implementation and would need their own check.

``aten::deg2rad`` has no ONNX symbolic function, which is why
[grade/model.py](../grade/model.py) multiplies by a constant instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from ..data import Vocabulary
from ..generator.model import BoulderGenerator, GeneratorConfig
from ..generator.tokenizer import GeneratorVocabulary
from ..grade.model import ModelConfig, TensionGradeTransformer

# The opset that onnxruntime-web supports widely at the time of writing.
OPSET = 17

PRECISIONS = ("fp32", "int8")


class CriticExport(nn.Module):
    """The critic with a float mask, so the web side never has to send bools."""

    def __init__(self, model: TensionGradeTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hold_type_ids: Tensor,
        orientations: Tensor,
        roles: Tensor,
        coordinates: Tensor,
        mask: Tensor,
        angles: Tensor,
    ) -> Tensor:
        # Input order matches forward_batch, which is the contract the web featurizer mirrors.
        return self.model(hold_type_ids, orientations, roles, coordinates, mask > 0.5, angles)


def critic_example_inputs(
    vocabulary: Vocabulary, *, batch: int = 2, nodes: int = 10
) -> tuple[Tensor, ...]:
    hold_types = max(len(vocabulary.hold_type_to_index), 1)
    return (
        torch.randint(0, hold_types, (batch, nodes), dtype=torch.long),
        torch.randn(batch, nodes, 2),
        torch.randint(0, 4, (batch, nodes), dtype=torch.long),
        torch.rand(batch, nodes, 2),
        torch.ones(batch, nodes),
        torch.full((batch,), 40.0),
    )


CRITIC_INPUTS = (
    "hold_type_ids",
    "orientations",
    "roles",
    "coordinates",
    "mask",
    "angles",
)


def export_critic(checkpoint: Path, destination: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    model = TensionGradeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    wrapper = CriticExport(model).eval()

    inputs = critic_example_inputs(vocabulary)
    dynamic_axes = {name: {0: "batch", 1: "nodes"} for name in CRITIC_INPUTS}
    dynamic_axes["angles"] = {0: "batch"}
    dynamic_axes["logits"] = {0: "batch"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(destination),
        input_names=list(CRITIC_INPUTS),
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )
    return {
        "model": "grade_critic",
        "path": str(destination),
        "inputs": list(CRITIC_INPUTS),
        "grade_labels": list(vocabulary.grade_labels),
        "temperature": float(payload.get("temperature", 1.0)),
    }


def export_generator(checkpoint: Path, destination: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocabulary = GeneratorVocabulary.from_dict(payload["vocabulary"])
    config = GeneratorConfig(**payload["model_config"])
    model = BoulderGenerator(config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    tokens = torch.randint(0, vocabulary.size, (2, 12), dtype=torch.long)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (tokens,),
        str(destination),
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch", 1: "length"}, "logits": {0: "batch", 1: "length"}},
        opset_version=OPSET,
        dynamo=False,
    )
    return {
        "model": "generator",
        "path": str(destination),
        "inputs": ["tokens"],
        "vocabulary_size": vocabulary.size,
        "max_sequence_length": config.max_sequence_length,
    }


def quantize(source: Path, destination: Path) -> None:
    """Dynamic int8 quantization, which roughly quarters the download."""

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", type=Path, default=Path("checkpoints/grade/tb2_12x12.pt"))
    parser.add_argument(
        "--generator", type=Path, default=Path("checkpoints/generator/tb2_12x12.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("web/public/models"))
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="int8",
        help="fp16 is deliberately unavailable; it overflows the attention mask constant",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for name, checkpoint, exporter in (
        ("grade.onnx", args.critic, export_critic),
        ("generator.onnx", args.generator, export_generator),
    ):
        if not checkpoint.exists():
            print(json.dumps({"skipped": str(checkpoint), "reason": "missing"}), flush=True)
            continue
        target = args.output / name
        summary = exporter(checkpoint, target)
        if args.precision == "int8":
            quantized = args.output / name.replace(".onnx", ".int8.onnx")
            quantize(target, quantized)
            summary["quantized_path"] = str(quantized)
            summary["quantized_bytes"] = quantized.stat().st_size
        summary["bytes"] = target.stat().st_size
        results.append(summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
