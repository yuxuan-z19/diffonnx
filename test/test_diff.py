import os
import subprocess

import onnx
import pytest
from grakel import Propagation, RandomWalkLabeled, ShortestPath
from onnx import ModelProto, TensorProto

import onnxdiff
from onnxdiff import *

from .models import get_programs


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
        providers=["CUDAExecutionProvider"],
        profile_dir=str(profile_dir),
        verbose=True,
    )
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.profiles) > 0

    assert profile_dir.exists()
    assert any(profile_dir.iterdir())

    files = list(profile_dir.glob("*.json"))
    assert len(files) == 2, "Expected profiling output for both models"
    for model_name in ["modelA", "modelB"]:
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


def test_runtimediff_invalid_providers():
    # Test RuntimeDiff with invalid providers
    ref_program, usr_program = get_programs()

    with pytest.raises(ValueError, match="Unsupported providers"):
        _ = RuntimeDiff(
            ref_program.model_proto,
            usr_program.model_proto,
            providers=["InvalidProvider"],
            verbose=True,
        )

    with pytest.raises(ValueError, match="Providers list cannot be empty"):
        _ = RuntimeDiff(
            ref_program.model_proto,
            usr_program.model_proto,
            providers=[],
            verbose=True,
        )
