from google._upb._message import Message, RepeatedCompositeContainer
from google.protobuf.json_format import MessageToDict
from onnx import ModelProto, GraphProto
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


def print_runtime_summary(result: RuntimeResult) -> None:
    print("Exact Match" if result.exact_match else "Not Exact Match")
    if result.equal:
        table = accuracy_table(result.equal)
        for t in table:
            t[0] = color(t[0], Status.Success)

        print(color("Equal Outputs:", Status.Success))
        print(
            tabulate(
                table,
                headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
                tablefmt="grid",
            )
        )
    if result.not_equal:
        table = accuracy_table(result.not_equal)
        for t in table:
            t[-2] = color(t[-2], Status.Warning)
            t[-1] = color(t[-1], Status.Warning)

        print(color("Not Equal Outputs:", Status.Warning))
        print(
            tabulate(
                table,
                headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
                tablefmt="grid",
            )
        )
    if result.mismatched:
        table = accuracy_table(result.mismatched)
        for t in table:
            if "\n" in t[1]:
                t[1] = color(t[1], Status.Error)
            if "\n" in t[2]:
                t[2] = color(t[2], Status.Error)

        print(color("Mismatched Outputs:", Status.Error))
        print(
            tabulate(
                table,
                headers=["Output", "Shape", "Dtype", "Cos Sim", "Max Error"],
                tablefmt="grid",
            )
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
    try:
        model_simplified, _ = simplify(model)
        if verbose:
            print("✅ ONNX model simplified successfully.")
        return model_simplified
    except Exception as e:
        if verbose:
            print(f"⚠️ ONNX simplification failed: {e}")
        return model
