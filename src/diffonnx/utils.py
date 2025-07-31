import importlib
import json
import os
import pathlib
import re
import sys
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from colorama import Fore
from colorama import init as colorama_init
from google._upb._message import Message, RepeatedCompositeContainer
from google.protobuf.json_format import MessageToDict
from onnx import GraphProto
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
            return json.loads(item).get("name", "")
        except:
            return item
    else:
        subitem_name = [get_name(sub_item) for sub_item in item]
        subitem_name = [name for name in subitem_name if name]
        return ", ".join(subitem_name) if subitem_name else None


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
    table = [[k, v] for k, v in result.score.items()]
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
                "\n".join(
                    name
                    for item in matches.a_diff
                    if (name := get_name(item)) is not None
                ),
                "\n".join(
                    name
                    for item in matches.b_diff
                    if (name := get_name(item)) is not None
                ),
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
                "Label",
                "Input Type Shape",
                "Output Type Shape",
                "A Duration (μs) / Name",
                "B Duration (μs) / Name",
            ],
            status=Status.Success,
        )


def get_graph_score_emb(result: StaticResult) -> np.ndarray:
    return np.array(list(result.score.values()))


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


def parse_node_irs(graph: GraphProto) -> Dict[str, str]:
    return {
        node.name: next(
            (p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.fx_node"),
            "",
        )
        for node in graph.node
        if any(p.key == "pkg.torch.onnx.fx_node" for p in node.metadata_props)
    }


def extract_kern_name(raw: str) -> Optional[str]:
    match = re.search(r"node_[^_/]+_\d+", raw)
    return match.group() if match else None


def parse_ort_profile(path: str, node_irs: Dict[str, str]) -> List[Profile]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"[ORTProfile] Failed to load '{path}': {e}")

    profiles = []
    op_count = defaultdict(int)

    nodes = sorted(
        (n for n in data if n.get("cat") == "Node"),
        key=lambda n: n.get("args", {}).get("node_index", 0),
    )
    node_kern_names = set(node_irs.keys())

    for node in nodes:
        args: Dict[str, Any] = node.get("args", {})
        op_name = args.get("op_name")
        if not op_name:
            continue

        idx = op_count[op_name]
        op_count[op_name] += 1

        kern_name = extract_kern_name(node.get("name", ""))
        kern_ir = node_irs.get(kern_name)
        if kern_ir is None:
            raise ValueError(
                f"[ORTProfile] Kernel name '{kern_name}' not found in node_irs {node_kern_names}.\n"
            )

        profiles.append(
            Profile(
                inst_label=f"{op_name}_{idx}",
                input_type_shape=args.get("input_type_shape", []),
                output_type_shape=args.get("output_type_shape", []),
                op_name0=kern_name,
                dur0=node.get("dur", -1),
                ir0=kern_ir,
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

    def _safe_get(obj, attr: str, default=None):
        return getattr(obj, attr, default) if obj else default

    a_map = {_key(p): p for p in profiles_a}
    b_map = {_key(p): p for p in profiles_b}
    all_keys = set(a_map.keys()) | set(b_map.keys())

    result: List[Profile] = []

    for k in all_keys:
        pa = a_map.get(k)
        pb = b_map.get(k)

        inst_label = pa.inst_label if pa else pb.inst_label
        input_type_shape = pa.input_type_shape if pa else pb.input_type_shape
        output_type_shape = pa.output_type_shape if pa else pb.output_type_shape

        result.append(
            Profile(
                inst_label=inst_label,
                input_type_shape=input_type_shape,
                output_type_shape=output_type_shape,
                op_name0=_safe_get(pa, "op_name0"),
                dur0=_safe_get(pa, "dur0", -1),
                ir0=_safe_get(pa, "ir0"),
                op_name1=_safe_get(pb, "op_name0"),
                dur1=_safe_get(pb, "dur0", -1),
                ir1=_safe_get(pb, "ir0"),
            )
        )

    return sorted(result, key=lambda p: p.inst_label)


def _patch_cudnn_ld_lib_path():
    if os.environ.get("DIFFONNX_PATCHED") == "1":
        return

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return

    nvidia_path = pathlib.Path(spec.submodule_search_locations[0])
    cudnn_lib = nvidia_path / "cudnn" / "lib"
    if not cudnn_lib.exists():
        return

    cudnn_lib_str = str(cudnn_lib.resolve())
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib_str not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_str}:{old_ld}"
        os.environ["DIFFONNX_PATCHED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
