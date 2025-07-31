import os
import subprocess

import onnx
import pytest
from grakel import Propagation, RandomWalkLabeled, ShortestPath
from onnx import ModelProto, TensorProto

import diffonnx
from diffonnx import *

from .common import get_programs


def test_staticdiff():
    # Static analysis check
    ref_program, usr_program = get_programs()

    diff = StaticDiff(ref_program.model_proto, usr_program.model_proto)
    results = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.score) == len(diff.graphdiff)

    parent = MainDiff(
        usr_program.model_proto,
        usr_program.model_proto,
        verbose=True,
    )
    diff = parent.static
    results = diff.summary(output=True)
    assert results.exact_match is True


def test_graphdiff():
    ref_program, usr_program = get_programs()

    def get_scores(diff_obj: StaticDiff) -> Dict[str, float]:
        return diff_obj.summary(output=True).score

    diff = StaticDiff(ref_program.model_proto, usr_program.model_proto)
    diff.graphdiff.add_kernels(
        [
            ShortestPath(normalize=True, with_labels=False),
            RandomWalkLabeled(normalize=True),
        ]
    )
    score = get_scores(diff)
    assert "ShortestPath" in score and "RandomWalkLabeled" in score

    diff = MainDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        verbose=True,
    )
    diff.static.graphdiff.remove_kernels(Propagation(normalize=True, random_state=42))
    score = get_scores(diff.static)
    assert "Propagation" not in score

    grakels = [
        Propagation(normalize=True),
        ShortestPath(normalize=True, with_labels=False),
    ]
    graphdiff = diffonnx.static.GraphDiff(grakels, verbose=True)
    diff = StaticDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        graphdiff=graphdiff,
    )
    score = get_scores(diff)

    assert len(graphdiff) == 2
    assert len(score) == 2


def test_runtimediff():
    # Runtime analysis check
    ref_program, usr_program = get_programs()

    diff = RuntimeDiff(ref_program.model_proto, usr_program.model_proto, verbose=True)
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.equal) == 0
    assert len(results.nonequal) != 0
    assert len(results.mismatched) == 0
    assert len(results.profiles) == 0

    parent = MainDiff(
        usr_program.model_proto,
        usr_program.model_proto,
        verbose=True,
    )
    diff = parent.runtime
    results = diff.summary(output=True)
    assert results.exact_match is True
    assert len(results.equal) != 0
    assert len(results.nonequal) == 0
    assert len(results.mismatched) == 0
    assert len(results.profiles) == 0


@pytest.mark.gpu
def test_runtimediff_gpu():
    # Runtime analysis check
    os.environ["DIFFONNX_PATCHED"] = "1"

    ref_program, usr_program = get_programs()

    diff = RuntimeDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        providers=["CUDAExecutionProvider"],
        verbose=True,
    )
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.equal) == 0
    assert len(results.nonequal) != 0
    assert len(results.mismatched) == 0
    assert len(results.profiles) == 0

    parent = MainDiff(
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
    assert len(results.profiles) == 0


def test_runtimediff_profiling(tmp_path):
    # Runtime profiling check
    ref_program, usr_program = get_programs()

    profile_dir = tmp_path / "log"
    diff = RuntimeDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        profile_dir=str(profile_dir),
        verbose=True,
    )
    results: RuntimeResult = diff.summary(output=True)
    assert results.exact_match is False
    assert len(results.profiles) > 0

    for p in results.profiles:
        if p.op_name0:
            assert p.dur0 >= 0
            assert p.ir0
        else:
            assert p.dur0 == -1
            assert p.ir0 is None

        if p.op_name1:
            assert p.dur1 != -1
            assert p.ir1
        else:
            assert p.dur1 == -1
            assert p.ir1 is None

    assert profile_dir.exists()
    assert any(profile_dir.iterdir())

    files = list(profile_dir.glob("*.json"))
    assert len(files) == 2, "Expected profiling output for both models"
    for model_name in ["modelA", "modelB"]:
        assert any(
            f.name.startswith(model_name) for f in files
        ), f"Profiling output for {model_name} not found"


@pytest.mark.gpu
def test_runtimediff_profiling_gpu(tmp_path):
    # Runtime profiling check
    ref_program, usr_program = get_programs()

    profile_dir = tmp_path / "log"
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

    for p in results.profiles:
        if p.op_name0:
            assert p.dur0 >= 0
            assert p.ir0
        else:
            assert p.dur0 == -1
            assert p.ir0 is None

        if p.op_name1:
            assert p.dur1 != -1
            assert p.ir1
        else:
            assert p.dur1 == -1
            assert p.ir1 is None

    assert profile_dir.exists()
    assert any(profile_dir.iterdir())

    files = list(profile_dir.glob("*.json"))
    assert len(files) == 2, "Expected profiling output for both models"
    for model_name in ["modelA", "modelB"]:
        assert any(
            f.name.startswith(model_name) for f in files
        ), f"Profiling output for {model_name} not found"


def test_diffonnx_cli(tmp_path):
    # CLI subprocess check
    ref_program, usr_program = get_programs()
    ref_path = os.path.join(tmp_path, "ref.onnx")
    usr_path = os.path.join(tmp_path, "usr.onnx")
    ref_program.save(ref_path)
    usr_program.save(usr_path)

    result = subprocess.run(
        ["diffonnx", ref_path, usr_path], capture_output=True, text=True
    )
    print("CLI stdout:", result.stdout)
    assert result.returncode == 0, f"diffonnx failed: {result.stderr}"
    assert "Not Exact Match" in result.stdout

    result = subprocess.run(
        ["diffonnx", usr_path, usr_path], capture_output=True, text=True
    )
    print("CLI stdout:", result.stdout)
    assert result.returncode == 0, f"diffonnx failed: {result.stderr}"
    assert "Exact Match" in result.stdout


def test_maindiff_invalid_inputs():
    # validate ONNX models
    ref_program, usr_program = get_programs()
    with pytest.raises(TypeError, match="onnx.ModelProto"):
        _ = MainDiff(ref_program.model, usr_program.model)

    ref_model = ref_program.model_proto
    null_model = ModelProto()

    with pytest.raises(ValueError, match="Empty or incomplete model.graph"):
        _ = MainDiff(ref_model, null_model)

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
        _ = MainDiff(invalid_model, invalid_model)


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

    _ = RuntimeDiff(
        ref_program.model_proto,
        usr_program.model_proto,
        providers=[],
        verbose=True,
    )
