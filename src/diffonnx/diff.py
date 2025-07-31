from typing import List, Optional, Tuple

from onnx import ModelProto

from .base import Diff
from .runtime import RuntimeDiff
from .static import GraphDiff, StaticDiff
from .structs import *


class MainDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        graphdiff: Optional[GraphDiff] = None,
        providers: Optional[List[str]] = None,
        profile_dir: Optional[str] = None,
        num_warmup: int = 3,
        verbose: bool = False,
    ):
        super().__init__(
            model_a=model_a,
            model_b=model_b,
            verbose=verbose,
        )

        self.static = StaticDiff(
            model_a=self.model_a,
            model_b=self.model_b,
            graphdiff=graphdiff,
            verbose=self._verbose,
        )
        self.runtime = RuntimeDiff(
            model_a=self.model_a,
            model_b=self.model_b,
            providers=providers,
            profile_dir=profile_dir,
            num_warmup=num_warmup,
            verbose=self._verbose,
        )

    def summary(self, output=False) -> Tuple[StaticResult, RuntimeResult]:
        static_sum = self.static.summary(output=output)
        runtime_sum = self.runtime.summary(output=output)
        return static_sum, runtime_sum
