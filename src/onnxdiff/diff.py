from onnx import ModelProto
from .structs import *

from typing import List, Tuple, Optional

from .base import Diff
from .static import StaticDiff, GraphDiff
from .runtime import RuntimeDiff


class OnnxDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        graphdiff: Optional[GraphDiff] = None,
        providers: Optional[List[str]] = None,
        profile_dir: Optional[str] = None,
        num_warmup: int = 10,
        verbose: bool = False,
    ):
        super().__init__(
            model_a=model_a,
            model_b=model_b,
            verbose=verbose,
        )

        self.static = StaticDiff(
            model_a=self._model_a,
            model_b=self._model_b,
            graphdiff=graphdiff,
            is_simplified=True,
            verbose=self._verbose,
        )
        self.runtime = RuntimeDiff(
            model_a=self._model_a,
            model_b=self._model_b,
            providers=providers,
            is_simplified=True,
            profile_dir=profile_dir,
            num_warmup=num_warmup,
            verbose=self._verbose,
        )

    def summary(self, output=False) -> Tuple[StaticResult, RuntimeResult]:
        static_sum = self.static.summary(output=output)
        runtime_sum = self.runtime.summary(output=output)
        return static_sum, runtime_sum
