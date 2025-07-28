import onnx
from onnx import ModelProto, GraphProto
import onnxruntime as ort
import numpy as np

from .base import Diff
from .utils import get_accuracy, print_runtime_summary
from .structs import *

from typing import List, Tuple


class RuntimeDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        providers: List[str] = None,
        is_simplified: bool = False,
        verbose: bool = False,
    ):
        super().__init__(model_a, model_b, verbose=verbose, is_simplified=is_simplified)
        default_provider = ["CPUExecutionProvider"]
        if providers:
            self.__check(providers)
            self.providers = providers + default_provider
        else:
            self.providers = default_provider

    def __check(self, providers: List[str]) -> None:
        assert set(ort.get_all_providers()).issuperset(
            providers
        ), f"ONNX Runtime does not support the following providers: {set(providers) - set(ort.get_all_providers())}"

    def _gen_inputs(
        self, graph: GraphProto, mod: int = -1, seed: int = 33550336
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)

        input_dict = {}
        for inp in graph.input:
            name = inp.name
            proto = inp.type.tensor_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(proto.elem_type)
            shape = [
                d.dim_value if d.HasField("dim_value") else 1 for d in proto.shape.dim
            ]

            match mod:
                case 0:
                    val = np.zeros(shape, dtype=dtype)
                case 1:
                    val = np.ones(shape, dtype=dtype)
                case _:
                    val = rng.random(shape).astype(dtype)

            input_dict[name] = val

        return input_dict

    def _compare_ndarrays(
        self,
        dict_a: Dict[str, np.ndarray],
        dict_b: Dict[str, np.ndarray],
        tol: float = 1e-6,
    ) -> Tuple[
        Dict[str, Tuple[np.ndarray, np.ndarray]],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
    ]:
        equal = {}
        non_equal = {}
        mismatched = {}

        keys = set(dict_a.keys()) | set(dict_b.keys())

        for key in keys:
            a = dict_a.get(key)
            b = dict_b.get(key)

            if a is None:
                mismatched[key + "_B"] = (np.zeros_like(b), b)
            elif b is None:
                mismatched[key + "_A"] = (a, np.zeros_like(a))
            elif a.shape != b.shape or a.dtype != b.dtype:
                mismatched[key] = (a, b)
            elif not np.allclose(a, b, rtol=tol, atol=tol):
                non_equal[key] = (a, b)
            else:
                equal[key] = (a, b)

        return equal, non_equal, mismatched

    def _infer(
        self, model: ModelProto, input_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        sess = ort.InferenceSession(model.SerializeToString(), providers=self.providers)
        output_names = [output.name for output in sess.get_outputs()]
        outputs = sess.run(output_names, input_dict)
        return {name: value for name, value in zip(output_names, outputs)}

    def _execute(self, mod: int = -1, seed: int = 33550336, tol: float = 1e-6):
        input_a = self._gen_inputs(self._model_a.graph, mod=mod, seed=seed)
        input_b = self._gen_inputs(self._model_b.graph, mod=mod, seed=seed)

        _, _, in_invalid = self._compare_ndarrays(input_a, input_b, tol=tol)
        if len(in_invalid) and len(in_invalid) > 0:
            print("⚠️ Input tensors have mismatched shapes or dtypes.")
            print(f"Mismatched keys: {list(in_invalid.keys())}")

        outputs_a = self._infer(self._model_a, input_a)
        outputs_b = self._infer(self._model_b, input_b)

        out_equal, out_nonequal, out_mismatched = self._compare_ndarrays(
            outputs_a, outputs_b, tol=tol
        )

        exact_match = True
        if out_mismatched or out_nonequal:
            exact_match = False
            if self._verbose:
                print("❌ Models are not exactly the same.")
                if out_mismatched:
                    print(f"⚠️ Shape/dtype mismatch keys: {list(out_mismatched.keys())}")
                if out_nonequal:
                    print(
                        f"⚠️ Value mismatch keys (within shape/dtype matched): {list(out_nonequal.keys())}"
                    )

        return exact_match, in_invalid, out_equal, out_nonequal, out_mismatched

    def summary(
        self,
        output: bool = False,
        mod: int = -1,
        seed: int = 33550336,
        tol: float = 1e-6,
    ) -> RuntimeResult:
        exact_match, invalid, equal, nonequal, mismatched = (
            self._execute(mod=mod, seed=seed, tol=tol)
        )

        result = RuntimeResult(
            exact_match=exact_match,
            invalid=get_accuracy(invalid),
            equal=get_accuracy(equal),
            nonequal=get_accuracy(nonequal),
            mismatched=get_accuracy(mismatched),
        )

        if output:
            print_runtime_summary(result)

        return result
