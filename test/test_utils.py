import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from diffonnx.structs import Profile
from diffonnx.utils import cos_sim_score, parse_node_irs, parse_ort_profile

from .common import get_programs


def test_cos_sim_score():
    torch.manual_seed(33550336)
    for _ in range(100):
        shape = torch.randint(2, 1024, (1,)).tolist()
        a = torch.rand(*shape)
        b = torch.rand(*shape)
        golden = torch.cosine_similarity(a, b, dim=0).item()
        score_ab: float = cos_sim_score(a.numpy(), b.numpy())
        score_ba: float = cos_sim_score(b.numpy(), a.numpy())
        assert (
            abs(score_ab - score_ba) < 1e-6
        ), f"cos_sim(a, b) ({score_ab}) != cos_sim(b, a) ({score_ba}) for shape {shape}"
        assert (
            abs(golden - score_ab) < 1e-6
        ), f"Expected cos_sim(a, b) = {golden}, got {score_ab} for shape {shape}"


def test_ort_profile(tmp_path):
    # onnxruntime profiling
    rng = np.random.default_rng(496)

    ref_program, _ = get_programs()
    ref_model = ref_program.model_proto
    ref_node_irs = parse_node_irs(ref_model.graph)

    input_dict = {}
    for inp in ref_model.graph.input:
        name = inp.name
        proto = inp.type.tensor_type
        dtype = onnx.helper.tensor_dtype_to_np_dtype(proto.elem_type)
        shape = [d.dim_value if d.HasField("dim_value") else 1 for d in proto.shape.dim]
        input_dict[name] = rng.random(shape).astype(dtype)

    sess_opt = ort.SessionOptions()
    sess_opt.enable_profiling = True
    sess_opt.profile_file_prefix = str(tmp_path / "model")

    sess = ort.InferenceSession(
        ref_model.SerializeToString(),
        sess_options=sess_opt,
    )

    output_names = [output.name for output in sess.get_outputs()]
    _ = sess.run(output_names, input_dict)
    profile_path = sess.end_profiling()

    profile = parse_ort_profile(profile_path, ref_node_irs)

    assert isinstance(
        profile, list
    ), "Parsed profile should be a list of Profile objects"
    assert len(profile) > 0, "Profile should contain at least one entry"
    assert isinstance(profile[0], Profile), "Profile entries should be of type Profile"

    assert os.path.exists(profile_path), "Profile file should exist"
    data = json.load(open(profile_path, "r"))
    nodes = sorted(
        (entry for entry in data if entry.get("cat") == "Node"),
        key=lambda x: x.get("args", {}).get("node_index", 0),
    )
    for node in nodes:
        args = node.get("args", {})
        assert "provider" in args, f"Expected 'provider' in node args, got {args}"
        assert (
            args.get("provider") == "CPUExecutionProvider"
        ), f"Expected 'CPUExecutionProvider', got {args.get('provider')}"


@pytest.mark.gpu
def test_ort_profile_gpu(tmp_path):
    # onnxruntime profiling
    rng = np.random.default_rng(496)

    ref_program, _ = get_programs()
    ref_model = ref_program.model_proto
    ref_node_irs = parse_node_irs(ref_model.graph)

    input_dict = {}
    for inp in ref_model.graph.input:
        name = inp.name
        proto = inp.type.tensor_type
        dtype = onnx.helper.tensor_dtype_to_np_dtype(proto.elem_type)
        shape = [d.dim_value if d.HasField("dim_value") else 1 for d in proto.shape.dim]
        input_dict[name] = rng.random(shape).astype(dtype)

    sess_opt = ort.SessionOptions()
    sess_opt.enable_profiling = True
    sess_opt.profile_file_prefix = str(tmp_path / "model")

    sess = ort.InferenceSession(
        ref_model.SerializeToString(),
        providers=["CUDAExecutionProvider"],
        sess_options=sess_opt,
    )

    output_names = [output.name for output in sess.get_outputs()]
    _ = sess.run(output_names, input_dict)
    profile_path = sess.end_profiling()

    profile = parse_ort_profile(profile_path, ref_node_irs)

    assert isinstance(
        profile, list
    ), "Parsed profile should be a list of Profile objects"
    assert len(profile) > 0, "Profile should contain at least one entry"
    assert isinstance(profile[0], Profile), "Profile entries should be of type Profile"

    assert os.path.exists(profile_path), "Profile file should exist"

    data = json.load(open(profile_path, "r"))
    nodes = sorted(
        (entry for entry in data if entry.get("cat") == "Node"),
        key=lambda x: x.get("args", {}).get("node_index", 0),
    )
    for node in nodes:
        args = node.get("args", {})
        assert "provider" in args, f"Expected 'provider' in node args, got {args}"
        assert (
            args.get("provider") == "CUDAExecutionProvider"
        ), f"Expected 'CUDAExecutionProvider', got {args.get('provider')}"
