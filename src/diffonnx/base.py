from abc import ABC, abstractmethod

import onnx


class Diff(ABC):
    def __init__(
        self,
        model_a: onnx.ModelProto,
        model_b: onnx.ModelProto,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self._check_model(model_a)
        self._check_model(model_b)

        self.model_a = model_a
        self.model_b = model_b

    def _check_model(self, model: onnx.ModelProto):
        if not isinstance(model, onnx.ModelProto) or not isinstance(
            model.graph, onnx.GraphProto
        ):
            raise TypeError(
                f"Expected onnx.ModelProto with a valid graph, got {type(model)} and {type(getattr(model, 'graph', None))}"
            )

        if (
            len(model.graph.node) == 0
            or len(model.graph.input) == 0
            or len(model.graph.output) == 0
        ):
            raise ValueError(
                f"Empty or incomplete model.graph: "
                f"nodes={len(model.graph.node)}, "
                f"inputs={len(model.graph.input)}, "
                f"outputs={len(model.graph.output)}"
            )

        try:
            onnx.checker.check_model(model)
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")

    @abstractmethod
    def summary(self, output=False):
        raise NotImplementedError("Subclasses must implement this method.")
