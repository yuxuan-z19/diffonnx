import onnx
from onnx import ModelProto, GraphProto
import onnxruntime as ort
import numpy as np

from .utils import get_accuracy, print_runtime_summary
from .structs import *

from typing import List, Tuple


class RuntimeDiff:
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        verbose: bool = False,
        providers: List[str] = None,
    ):
        self._model_a = model_a
        self._model_b = model_b
        self._verbose = verbose

        self.providers = ["CPUExecutionProvider"]
        if providers:
            self.__check(providers)
            self.providers = providers + self.providers

    def __check(self, providers: List[str]) -> None:
        assert set(ort.get_all_providers()).issuperset(
            providers
        ), f"ONNX Runtime does not support the following providers: {set(providers) - set(ort.get_all_providers())}"

    def _gen_inputs(
        self, graph: GraphProto, mod: int = -1, seed: int = 33550336
    ) -> Dict[str, np.ndarray]:
        np.random.seed(seed)

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
                    val = np.random.random(shape).astype(dtype)

            input_dict[name] = val

        return input_dict

    def _check_ndarrays(
        self,
        dict_a: Dict[str, np.ndarray],
        dict_b: Dict[str, np.ndarray],
        check_val: bool = False,
        tol: float = 1e-6,
    ):
        keys_a = set(dict_a.keys())
        keys_b = set(dict_b.keys())
        if keys_a != keys_b:
            miss_a = keys_a - keys_b
            miss_b = keys_b - keys_a
            raise KeyError(
                f"Input tensors for models A and B do not match. Missing in A: {miss_a}, Missing in B: {miss_b}"
            )

        for key in keys_a:
            a, b = dict_a[key], dict_b[key]
            if a.shape != b.shape:
                raise ValueError(
                    f"Input tensor '{key}' has different shapes in models A and B. "
                    f"Shape A: {a.shape}, Shape B: {b.shape}"
                )
            elif a.dtype != b.dtype:
                raise ValueError(
                    f"Input tensor '{key}' has different dtypes in models A and B. "
                    f"Dtype A: {a.dtype}, Dtype B: {b.dtype}"
                )
            elif check_val and not np.allclose(a, b, rtol=tol, atol=tol):
                raise ValueError(
                    f"Input tensor '{key}' has different values in models A and B. "
                    f"Values A: {a}, Values B: {b}"
                )

        print("✅ All input names, shapes, and dtypes match.")

    def _compare_ndarrays(
        self,
        dict_a: Dict[str, np.ndarray],
        dict_b: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, Tuple[np.ndarray, np.ndarray]],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
    ]:
        matched: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        mismatched: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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
            else:
                matched[key] = (a, b)

        return matched, mismatched

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
        self._check_ndarrays(input_a, input_b)

        outputs_a = self._infer(self._model_a, input_a)
        outputs_b = self._infer(self._model_b, input_b)

        matched, mismatched = self._compare_ndarrays(outputs_a, outputs_b)

        equal = {}
        not_equal = {}
        for key, (a, b) in matched.items():
            if np.allclose(a, b, rtol=tol, atol=tol):
                equal[key] = (a, b)
            else:
                not_equal[key] = (a, b)

        exact_match = True
        if mismatched or not_equal:
            exact_match = False
            if self._verbose:
                print("❌ Models are not exactly the same.")
                if mismatched:
                    print(f"⚠️ Shape/dtype mismatch keys: {list(mismatched.keys())}")
                if not_equal:
                    print(
                        f"⚠️ Value mismatch keys (within shape/dtype matched): {list(not_equal.keys())}"
                    )

        return exact_match, equal, not_equal, mismatched

    def summary(
        self,
        output: bool = False,
        mod: int = -1,
        seed: int = 33550336,
        tol: float = 1e-6,
    ) -> RuntimeResult:
        exact_match, equal, not_equal, mismatched = self._execute(
            mod=mod, seed=seed, tol=tol
        )

        result = RuntimeResult(
            exact_match=exact_match,
            equal=get_accuracy(equal),
            not_equal=get_accuracy(not_equal),
            mismatched=get_accuracy(mismatched),
        )

        if output:
            print_runtime_summary(result)

        return result
