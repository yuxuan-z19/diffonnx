from onnx import ModelProto
from abc import ABC, abstractmethod
from .utils import try_simplify


class Diff(ABC):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        is_simplified: bool = False,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self._model_a = self._prepare_model(model_a, is_simplified)
        self._model_b = self._prepare_model(model_b, is_simplified)

    def _prepare_model(self, model: ModelProto, is_simplified: bool) -> ModelProto:
        return model if is_simplified else try_simplify(model, verbose=self._verbose)

    @abstractmethod
    def summary(self, output=False):
        raise NotImplementedError("Subclasses must implement this method.")
