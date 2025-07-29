import onnx
from onnx import ModelProto, GraphProto
from google._upb._message import Message, RepeatedCompositeContainer
from google.protobuf.json_format import MessageToDict
from onnxsim import simplify
import json

from enum import Enum
from tabulate import tabulate
from colorama import init as colorama_init
from colorama import Fore
import numpy as np
from typing import List, Union, Tuple, Dict

from .structs import *

colorama_init()


class Status(Enum):
    Success = 0
    Warning = 1
    Error = 2


color_map = {
    Status.Success: Fore.GREEN,
    Status.Warning: Fore.YELLOW,
    Status.Error: Fore.RED,
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


def print_static_summary(result: StaticResult) -> None:
    # top line
    print("Exact Match" if result.exact_match else "Not Exact Match")

    # score
    table = [[k, v] for k, v in result.score.graph_kernel_scores.items()]
    print(
        tabulate(
            table,
            headers=["Kernels", "Score"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        )
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

    print(
        tabulate(
            count_list,
            headers=["Fields", "A", "B"],
            tablefmt="rounded_outline",
        )
    )

    print(
        tabulate(
            diff_list,
            headers=["Fields", "A Diff", "B Diff"],
            tablefmt="grid",
        )
    )


def _print_colored_table(
    title: str,
    rows,
    status: Status,
    headers,
    color_indices: list[int] = None,
    color_entire_row: bool = False,
):
    for t in rows:
        if color_entire_row:
            t[0] = color(t[0], status)
        if color_indices:
            for idx in color_indices:
                t[idx] = color(t[idx], status)

    print(title)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_runtime_summary(result: RuntimeResult) -> None:
    print("Exact Match" if result.exact_match else "Not Exact Match")

    if result.invalid:
        _print_colored_table(
            "Invalid Inputs:",
            accuracy_table(result.invalid),
            Status.Error,
            headers=["Input", "Shape", "Dtype", "Cos Sim", "Max Error"],
            color_indices=[1, 2],
        )

    if result.equal:
        _print_colored_table(
            "Equal Outputs:",
            accuracy_table(result.equal),
            Status.Success,
            headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            color_entire_row=True,
        )

    if result.nonequal:
        _print_colored_table(
            "Not Equal Outputs:",
            accuracy_table(result.nonequal),
            Status.Warning,
            headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            color_indices=[-2, -1],
        )

    if result.mismatched:
        _print_colored_table(
            "Mismatched Outputs:",
            accuracy_table(result.mismatched),
            Status.Error,
            headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
            color_indices=[1, 2],
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
