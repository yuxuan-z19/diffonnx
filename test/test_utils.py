import torch
import torch.nn as nn
from onnxdiff.utils import cos_sim_score, try_simplify
import onnx
from onnx import ModelProto, TensorProto

import pytest


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


def test_try_simplify():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    model_program = torch.onnx.export(model, torch.randn(1, 10), dynamo=True)
    simplified_model = try_simplify(model_program.model_proto)
    assert isinstance(
        simplified_model, ModelProto
    ), "Simplified model should be an ONNX ModelProto"

    with pytest.raises(TypeError, match="Expected onnx.ModelProto with a valid graph"):
        _ = try_simplify(model_program)

    null_model = ModelProto()
    with pytest.raises(ValueError, match="Empty or incomplete model.graph"):
        _ = try_simplify(null_model)

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
        _ = try_simplify(invalid_model)
