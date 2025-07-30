import json
from collections import defaultdict
from enum import Enum
from os import name
from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from colorama import Fore
from colorama import init as colorama_init
from google._upb._message import Message, RepeatedCompositeContainer
from google.protobuf.json_format import MessageToDict
from onnx import GraphProto, ModelProto
from onnxsim import simplify
from sympy import O
from tabulate import tabulate

from .structs import *

colorama_init()


class Status(Enum):
    Success = 0
    Warning = 1
    Error = 2
    Highlight = 3


color_map = {
    Status.Success: Fore.GREEN,
    Status.Warning: Fore.YELLOW,
    Status.Error: Fore.RED,
    Status.Highlight: Fore.CYAN,
}


def color(text: str, status: Status) -> str:
    return f"{color_map[status]}{text}{Fore.RESET}"


def matches_string(count: int, total: int):
    text = f"{count}/{total}"
    status = Status.Success if count == total else Status.Error
    return color(text=text, status=status)


def accuracy_table(accuracy: Dict[str, Accuracy]) -> List[List[str]]:
    table = []
    for key, acc in accuracy.items():
        table.append(
            [key, acc.shape, acc.dtype, f"{acc.cos_sim:.4f}", f"{acc.max_err:.4f}"]
        )
    return table


def get_name(item):
    if isinstance(item, str):
        try:
            return json.loads(item).get("name", str(item))
        except:
            return item
    else:
        return ", ".join(get_name(sub_item) for sub_item in item)


def hashmsg(msg: Message) -> str:
    if isinstance(msg, Message):
        d = MessageToDict(msg, preserving_proto_field_name=True)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))
    else:
        return str(msg)


def hashitem(items: List[Union[Message, RepeatedCompositeContainer]]) -> frozenset:
    item_set = set()
    for item in items:
        if isinstance(item, RepeatedCompositeContainer):
            frozen = [hashmsg(sub_item) for sub_item in item]
            item_set.add(frozenset(frozen))
        else:
            item_set.add(hashmsg(item))
    return item_set


def _print_colored_table(
    title: str,
    rows,
    headers,
    status: Status = Status.Highlight,
    color_indices: list[int] = None,
    color_entire_row: bool = False,
    floatfmt: str = "",
):
    for t in rows:
        if color_entire_row:
            t[0] = color(t[0], status)
        if color_indices:
            for idx in color_indices:
                t[idx] = color(t[idx], status)

    print(title)
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=floatfmt))


def print_static_summary(result: StaticResult) -> None:
    # top line
    print("Exact Match" if result.exact_match else "Not Exact Match")

    # score
    table = [[k, v] for k, v in result.score.graph_kernel_scores.items()]
    _print_colored_table(
        "Graph Kernel Scores",
        table,
        ["Kernels", "Score"],
        floatfmt=".4f",
    )

    # differences
    count_list = []
    diff_list = []

    for key, matches in result.graph_matches.items():
        field = f"Graph.{key.capitalize()}"
        count_list.append(
            [
                field,
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )

        diff_list.append(
            [
                field,
                "\n".join(get_name(item) for item in matches.a_diff),
                "\n".join(get_name(item) for item in matches.b_diff),
            ]
        )

    for key, matches in result.root_matches.items():
        count_list.append(
            [
                f"{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )

    _print_colored_table(
        "Matched Fields Count", count_list, headers=["Fields", "A", "B"]
    )

    # 输出 Diff 表格
    _print_colored_table(
        "Mismatched Items",
        diff_list,
        ["Fields", "A Diff", "B Diff"],
    )


def print_runtime_summary(result: RuntimeResult) -> None:
    print("Exact Match" if result.exact_match else "Not Exact Match")

    if result.invalid:
        _print_colored_table(
            "Invalid Inputs:",
            accuracy_table(result.invalid),
            ["Input", "Shape", "Dtype", "Cos Sim", "Max Error"],
            status=Status.Error,
            color_indices=[1, 2],
        )

    if result.equal:
        _print_colored_table(
            "Equal Outputs:",
            accuracy_table(result.equal),
            ["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            status=Status.Success,
            color_entire_row=True,
        )

    if result.nonequal:
        _print_colored_table(
            "Not Equal Outputs:",
            accuracy_table(result.nonequal),
            ["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            status=Status.Warning,
            color_indices=[-2, -1],
        )

    if result.mismatched:
        _print_colored_table(
            "Mismatched Outputs:",
            accuracy_table(result.mismatched),
            ["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            status=Status.Error,
            color_indices=[1, 2],
        )

    if result.profiles:
        max_a_idx = max_b_idx = 0
        max_a_val = max_b_val = float("-inf")
        for i, p in enumerate(result.profiles):
            if p.dur0 > max_a_val:
                max_a_idx, max_a_val = i, p.dur0
            if p.dur1 > max_b_val:
                max_b_idx, max_b_val = i, p.dur1

        color_a = Status.Warning
        color_b = Status.Highlight

        rows = []
        for i, p in enumerate(result.profiles):
            label = p.inst_label
            dur0 = str(p.dur0) if p.dur0 != -1 else "N/A"
            dur0 += f" ({str(p.op_name0)})" if p.op_name0 else ""
            dur1 = str(p.dur1) if p.dur1 != -1 else "N/A"
            dur1 += f" ({str(p.op_name1)})" if p.op_name1 else ""

            if i == max_a_idx and i == max_b_idx:
                label = color(label, Status.Highlight)
                dur0 = color(dur0, Status.Warning)
                dur1 = color(dur1, Status.Highlight)
            else:
                if i == max_a_idx:
                    label = color(label, color_a)
                    dur0 = color(dur0, color_a)
                if i == max_b_idx:
                    label = color(label, color_b)
                    dur1 = color(dur1, color_b)

            rows.append(
                [
                    label,
                    json.dumps(p.input_type_shape, separators=(",", ":")),
                    json.dumps(p.output_type_shape, separators=(",", ":")),
                    dur0,
                    dur1,
                ]
            )

        _print_colored_table(
            "Profile Comparison:",
            rows,
            [
                "Name",
                "Input Type Shape",
                "Output Type Shape",
                "A Duration",
                "B Duration",
            ],
            status=Status.Success,
        )


def get_graph_score_emb(result: StaticResult) -> np.ndarray:
    return np.array(list(result.score.graph_kernel_scores.values()))


def cos_sim_score(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def ang_sim_score(a: np.ndarray, b: np.ndarray) -> float:
    score = cos_sim_score(a, b)
    score = np.clip(score, -1.0, 1.0)
    return 1 - (np.arccos(score) / np.pi)


def get_accuracy(result_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> dict:
    accuracy = {}
    for key, (a, b) in result_dict.items():
        shape = str(a.shape)
        dtype = str(a.dtype)
        if a.shape != b.shape:
            shape += "\n" + str(b.shape)
        if a.dtype != b.dtype:
            dtype += "\n" + str(b.dtype)

        cos_sim = cos_sim_score(a.flatten(), b.flatten())
        max_err = np.max(np.abs(a - b))
        accuracy[key] = Accuracy(
            shape=shape, dtype=dtype, cos_sim=cos_sim, max_err=max_err
        )
    return accuracy


def try_simplify(model: ModelProto, verbose: bool = False) -> ModelProto:
    if not isinstance(model, ModelProto) or not isinstance(model.graph, GraphProto):
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

    try:
        # `check_n=False` avoids shape checking that may fail in rare cases
        model_simplified, success = simplify(model, check_n=False)
        if not success:
            if verbose:
                print("⚠️ Simplification reported failure, returning original model.")
            return model
        if verbose:
            print("✅ ONNX model simplified successfully.")
        return model_simplified
    except Exception as e:
        if verbose:
            print(f"⚠️ ONNX simplification failed: {e}")
        return model


def parse_ort_profile(path: str) -> List[Profile]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Failed to read or parse file '{path}': {e}")

    profiles = []
    op_count = defaultdict(int)

    nodes = sorted(
        (entry for entry in data if entry.get("cat") == "Node"),
        key=lambda x: x.get("args", {}).get("node_index", 0),
    )

    for node in nodes:
        args = node.get("args", {})
        op_name = args.get("op_name")
        if op_name is None:
            continue

        idx = op_count[op_name]
        op_count[op_name] += 1

        raw_name: str = node.get("name", "")
        kernel_name = raw_name.removesuffix("_kernel_time") if raw_name else ""

        profiles.append(
            Profile(
                inst_label=f"{op_name}_{idx}",
                input_type_shape=args.get("input_type_shape", []),
                output_type_shape=args.get("output_type_shape", []),
                op_name0=kernel_name,
                dur0=node.get("dur", -1),
            )
        )

    return profiles


def get_profile_compares(
    profiles_a: List[Profile], profiles_b: List[Profile]
) -> List[Profile]:
    def _key(profile: Profile) -> Tuple:
        return (
            profile.inst_label,
            json.dumps(profile.input_type_shape, sort_keys=True),
            json.dumps(profile.output_type_shape, sort_keys=True),
        )

    a_map = {_key(p): p for p in profiles_a}
    b_map = {_key(p): p for p in profiles_b}
    all_keys = set(a_map.keys()) | set(b_map.keys())

    result: List[Profile] = []

    for k in all_keys:
        pa = a_map.get(k)
        pb = b_map.get(k)
        result.append(
            Profile(
                inst_label=pa.inst_label if pa else pb.inst_label,
                input_type_shape=pa.input_type_shape if pa else pb.input_type_shape,
                output_type_shape=(
                    pa.output_type_shape if pa else pb.output_type_shape
                ),
                op_name0=pa.op_name0 if pa else None,
                dur0=pa.dur0 if pa else -1,
                op_name1=pb.op_name0 if pb else None,
                dur1=pb.dur0 if pb else -1,
            )
        )

    return sorted(result, key=lambda p: p.inst_label)
