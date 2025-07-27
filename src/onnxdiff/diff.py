import onnx.checker
from onnx import ModelProto, GraphProto
from onnxsim import simplify
from .structs import *

from typing import List, Tuple

from .static import StaticDiff
from .runtime import RuntimeDiff
from .utils import try_simplify


class OnnxDiff:
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        verbose: bool = False,
        providers: List[str] = None,
    ):
        self.__check(model_a)
        self.__check(model_b)

        self._verbose = verbose
        self._model_a = try_simplify(model_a, verbose=self._verbose)
        self._model_b = try_simplify(model_b, verbose=self._verbose)

        self.static = StaticDiff(
            model_a=self._model_a, model_b=self._model_b, verbose=self._verbose
        )
        self.runtime = RuntimeDiff(
            model_a=self._model_a,
            model_b=self._model_b,
            verbose=self._verbose,
            providers=providers,
        )

    def __check(self, model):
        if not isinstance(model, ModelProto):
            raise TypeError(
                f"Model must be an instance of onnx.ModelProto, got: {str(type(model))}"
            )
        if not model.HasField("graph") or not isinstance(model.graph, GraphProto):
            raise ValueError(
                f"Model must contain a valid graph field of type onnx.GraphProto, got: {str(type(model.graph))}"
            )

        try:  # finally check the validity
            onnx.checker.check_model(model)
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")

    def summary(self, output=False) -> Tuple[StaticResult, RuntimeResult]:
        static_sum = self.static.summary(output=output)
        runtime_sum = self.runtime.summary(output=output)
        return static_sum, runtime_sum
