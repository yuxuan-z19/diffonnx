import os
from typing import List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import GraphProto, ModelProto

from .base import Diff
from .structs import *
from .utils import (
    _patch_cudnn_ld_lib_path,
    get_accuracy,
    get_profile_compares,
    parse_node_irs,
    parse_ort_profile,
    print_runtime_summary,
)

TensorMap = Dict[str, np.ndarray]
ORTConfig = str | Tuple[str, Dict[str, str]]


class RuntimeDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        providers: Optional[List[ORTConfig]] = None,
        profile_dir: Optional[str] = None,
        num_warmup: int = 3,
        verbose: bool = False,
    ):
        self._prepare_providers(providers, verbose=verbose)

        super().__init__(model_a, model_b, verbose=verbose)

        self._node_irs_a = parse_node_irs(model_a.graph)
        self._node_irs_b = parse_node_irs(model_b.graph)
        self._profiling = profile_dir is not None

        if self._profiling and not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)
        self._profile_dir = profile_dir

        self.num_warmup = num_warmup

        if self._profiling and self._verbose:
            print("<RuntimeDiff> Profiling enabled.")
            print(f"Node IRs for modelA: {self._node_irs_a.keys()}")
            print(f"Node IRs for modelB: {self._node_irs_b.keys()}")

    def _prepare_providers(
        self,
        providers: Optional[List[ORTConfig]] = None,
        verbose: bool = False,
    ) -> None:
        default_provider = ["CPUExecutionProvider"]
        if providers:
            available = set(ort.get_available_providers())
            names = [p if isinstance(p, str) else p[0] for p in providers]
            missing = set(names) - available
            if missing:
                raise ValueError(
                    f"Unsupported providers: {missing}. Available providers: {available}"
                )
            if "CUDAExecutionProvider" in providers:
                _patch_cudnn_ld_lib_path()
            self.providers = providers + default_provider
        else:
            if verbose:
                print(
                    "<RuntimeDiff> No providers specified, using default CPUExecutionProvider."
                )
            self.providers = default_provider

    def _gen_inputs(
        self, graph: GraphProto, mod: int = -1, seed: int = 33550336
    ) -> TensorMap:
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
        self, dict_a: TensorMap, dict_b: TensorMap, tol: float = 1e-6
    ) -> Tuple[Dict[str, OutputPair], Dict[str, OutputPair], Dict[str, OutputPair]]:
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
        self,
        model: ModelProto,
        input_dict: TensorMap,
        model_name: str = "model",
        profiling: bool = False,
        node_irs: Optional[Dict[str, str]] = None,
    ) -> Tuple[TensorMap, List[Profile]]:
        sess_opt = ort.SessionOptions()
        if profiling:
            sess_opt.enable_profiling = True
            sess_opt.profile_file_prefix = os.path.join(self._profile_dir, model_name)

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=self.providers, sess_options=sess_opt
        )

        output_names = [output.name for output in sess.get_outputs()]
        outputs = sess.run(output_names, input_dict)
        result = {name: value for name, value in zip(output_names, outputs)}

        if profiling:
            profile_path = sess.end_profiling()
            profile = parse_ort_profile(profile_path, node_irs=node_irs)
            if self._verbose:
                print(f"<RuntimeDiff> Profiling results saved to {profile_path}")
        else:
            profile = []

        return result, profile

    def _execute(self, mod: int = -1, seed: int = 33550336, tol: float = 1e-6):
        input_a = self._gen_inputs(self.model_a.graph, mod=mod, seed=seed)
        input_b = self._gen_inputs(self.model_b.graph, mod=mod, seed=seed)

        _, _, in_invalid = self._compare_ndarrays(input_a, input_b, tol=tol)
        if len(in_invalid) and len(in_invalid) > 0:
            print("<RuntimeDiff> ⚠️ Input tensors have mismatched shapes or dtypes.")
            print(f"Mismatched keys: {list(in_invalid.keys())}")

        for _ in range(self.num_warmup):
            _ = self._infer(self.model_a, input_a)
            _ = self._infer(self.model_b, input_b)

        outputs_a, profile_a = self._infer(
            self.model_a,
            input_a,
            model_name="modelA",
            profiling=self._profiling,
            node_irs=self._node_irs_a,
        )
        outputs_b, profile_b = self._infer(
            self.model_b,
            input_b,
            model_name="modelB",
            profiling=self._profiling,
            node_irs=self._node_irs_b,
        )

        out_equal, out_nonequal, out_mismatched = self._compare_ndarrays(
            outputs_a, outputs_b, tol=tol
        )

        exact_match = not (out_nonequal or out_mismatched)
        if not exact_match and self._verbose:
            print("<RuntimeDiff> ❌ Models are not exactly the same.")
            if out_mismatched:
                print(
                    f"<RuntimeDiff> ⚠️ Shape/dtype mismatch keys: {list(out_mismatched.keys())}"
                )
            if out_nonequal:
                print(
                    f"<RuntimeDiff> ⚠️ Value mismatch keys (within shape/dtype matched): {list(out_nonequal.keys())}"
                )

        return ExecutionStats(
            exact_match=exact_match,
            in_invalid=in_invalid,
            out_equal=out_equal,
            out_nonequal=out_nonequal,
            out_mismatched=out_mismatched,
            profile_a=profile_a,
            profile_b=profile_b,
        )

    def summary(
        self,
        output: bool = False,
        mod: int = -1,
        seed: int = 33550336,
        tol: float = 1e-6,
    ) -> RuntimeResult:
        exec_res = self._execute(mod=mod, seed=seed, tol=tol)

        result = RuntimeResult(
            exact_match=exec_res.exact_match,
            invalid=get_accuracy(exec_res.in_invalid),
            equal=get_accuracy(exec_res.out_equal),
            nonequal=get_accuracy(exec_res.out_nonequal),
            mismatched=get_accuracy(exec_res.out_mismatched),
            profiles=get_profile_compares(exec_res.profile_a, exec_res.profile_b),
        )

        if output:
            print_runtime_summary(result)

        return result
