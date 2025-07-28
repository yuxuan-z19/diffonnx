from onnx import ModelProto
from .structs import *

from typing import List, Tuple

from .base import Diff
from .static import StaticDiff
from .runtime import RuntimeDiff


class OnnxDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        verbose: bool = False,
        providers: List[str] = None,
    ):
        super().__init__(
            model_a=model_a,
            model_b=model_b,
            verbose=verbose,
        )

        self.static = StaticDiff(
            model_a=self._model_a,
            model_b=self._model_b,
            verbose=self._verbose,
            is_simplified=True,
        )
        self.runtime = RuntimeDiff(
            model_a=self._model_a,
            model_b=self._model_b,
            verbose=self._verbose,
            providers=providers,
            is_simplified=True,
        )

    def summary(self, output=False) -> Tuple[StaticResult, RuntimeResult]:
        static_sum = self.static.summary(output=output)
        runtime_sum = self.runtime.summary(output=output)
        return static_sum, runtime_sum
