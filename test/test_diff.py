import torch
import torch.nn as nn
from torch.onnx import ONNXProgram
import onnx
from onnx import ModelProto, TensorProto
import math
from grakel import ShortestPath, RandomWalkLabeled, Propagation


import os
import subprocess
from typing import Tuple
import pytest

import onnxdiff
from onnxdiff import *


# Reference: https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/86_Matmul_Divide_GELU.py
class Model(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x


# From: https://github.com/deepreinforce-ai/CUDA-L1
class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.divisor = divisor

        scaled_weight = (self.weight / divisor).detach().clone()
        scaled_bias = (self.bias / divisor).detach().clone()

        self.register_buffer("scaled_weight_t", scaled_weight.t().contiguous())
        self.register_buffer("scaled_bias", scaled_bias.contiguous())

    def forward(self, x):
        out = torch.addmm(self.scaled_bias, x, self.scaled_weight_t)
        return torch.nn.functional.gelu(out)


# === Helper functions ===
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]


def get_programs() -> Tuple[ONNXProgram, ONNXProgram]:
    ref = Model(*get_init_inputs())
    usr = ModelNew(*get_init_inputs())
    inputs = get_inputs()

    ref_program = torch.onnx.export(ref, tuple(inputs), dynamo=True)
    usr_program = torch.onnx.export(usr, tuple(inputs), dynamo=True)

    ref_program.optimize()
    usr_program.optimize()

    return ref_program, usr_program


# === Pytest test case ===
def test_static_diff():
    # Static analysis check
    ref_program, usr_program = get_programs()

    diff = StaticDiff(ref_program.model_proto, usr_program.model_proto)
    results = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.score.graph_kernel_scores) == len(diff.graphdiff)

    parent = OnnxDiff(
        usr_program.model_proto,
        usr_program.model_proto,
        providers=["CUDAExecutionProvider"],
        verbose=True,
    )
    diff = parent.static
    results = diff.summary(output=True)
    assert results.exact_match is True


def test_graph_diff():
    ref_program, usr_program = get_programs()

    def get_scores(diff_obj: StaticDiff) -> Dict[str, float]:
        return diff_obj.summary(output=True).score.graph_kernel_scores

    diff = StaticDiff(ref_program.model_proto, usr_program.model_proto)
    diff.graphdiff.add_kernels(
        [
            ShortestPath(normalize=True, with_labels=False),
            RandomWalkLabeled(normalize=True),
        ]
    )
    score = get_scores(diff)
    assert "ShortestPath" in score and "RandomWalkLabeled" in score

    diff = OnnxDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        providers=["CUDAExecutionProvider"],
        verbose=True,
    )
    diff.static.graphdiff.remove_kernels(Propagation(normalize=True, random_state=42))
    score = get_scores(diff.static)
    assert "Propagation" not in score

    grakels = [
        Propagation(normalize=True),
        ShortestPath(normalize=True, with_labels=False),
    ]
    graphdiff = onnxdiff.static.GraphDiff(grakels, verbose=True)
    diff = StaticDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        graphdiff=graphdiff,
    )
    score = get_scores(diff)

    assert len(graphdiff) == 2
    assert len(score) == 2


def test_runtime_diff():
    # Runtime analysis check
    ref_program, usr_program = get_programs()

    diff = RuntimeDiff(ref_program.model_proto, usr_program.model_proto, verbose=True)
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.equal) == 0
    assert len(results.nonequal) != 0
    assert len(results.mismatched) == 0

    parent = OnnxDiff(
        usr_program.model_proto,
        usr_program.model_proto,
        providers=["CUDAExecutionProvider"],
        verbose=True,
    )
    diff = parent.runtime
    results = diff.summary(output=True)
    assert results.exact_match is True
    assert len(results.equal) != 0
    assert len(results.nonequal) == 0
    assert len(results.mismatched) == 0


def test_runtime_diff_profiling(tmp_path):
    # Runtime profiling check
    ref_program, usr_program = get_programs()

    profile_dir = tmp_path / "profiling"
    diff = RuntimeDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        profile_dir=str(profile_dir),
        verbose=True,
    )
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.equal) == 0
    assert len(results.nonequal) != 0
    assert len(results.mismatched) == 0

    assert profile_dir.exists()
    assert any(profile_dir.iterdir())

    files = list(profile_dir.glob("*.json"))
    assert len(files) == 2, "Expected profiling output for both models"
    for model_name in ["model_a", "model_b"]:
        assert any(
            f.name.startswith(model_name) for f in files
        ), f"Profiling output for {model_name} not found"


def test_onnxdiff_cli(tmp_path):
    # CLI subprocess check
    ref_program, usr_program = get_programs()
    ref_path = os.path.join(tmp_path, "ref.onnx")
    usr_path = os.path.join(tmp_path, "usr.onnx")
    ref_program.save(ref_path)
    usr_program.save(usr_path)

    result = subprocess.run(
        ["onnxdiff", ref_path, usr_path], capture_output=True, text=True
    )
    print("CLI stdout:", result.stdout)
    assert result.returncode == 0, f"onnxdiff failed: {result.stderr}"
    assert "Not Exact Match" in result.stdout

    result = subprocess.run(
        ["onnxdiff", usr_path, usr_path], capture_output=True, text=True
    )
    print("CLI stdout:", result.stdout)
    assert result.returncode == 0, f"onnxdiff failed: {result.stderr}"
    assert "Exact Match" in result.stdout


def test_onnxdiff_invalid_inputs():
    # validate ONNX models
    ref_program, usr_program = get_programs()
    with pytest.raises(TypeError, match="onnx.ModelProto"):
        _ = OnnxDiff(ref_program.model, usr_program.model)

    ref_model = ref_program.model_proto
    null_model = ModelProto()

    with pytest.raises(ValueError, match="Empty or incomplete model.graph"):
        _ = OnnxDiff(ref_model, null_model)

    def _gen_invalid_model() -> ModelProto:
        node = onnx.helper.make_node(
            "NonExistOp",
            inputs=["X"],
            outputs=["Y"],
        )
        graph = onnx.helper.make_graph(
            nodes=[node],
            name="InvalidOpGraph",
            inputs=[onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            outputs=[onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = onnx.helper.make_model(graph)
        return model

    invalid_model = _gen_invalid_model()
    with pytest.raises(ValueError, match="Invalid ONNX model"):
        _ = OnnxDiff(invalid_model, invalid_model)
