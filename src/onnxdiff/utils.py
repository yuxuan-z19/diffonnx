from google._upb._message import Message, RepeatedCompositeContainer
from google.protobuf.json_format import MessageToDict
import json

from enum import Enum
from tabulate import tabulate
from colorama import init as colorama_init
from colorama import Fore
import numpy as np
from typing import List, Union

from .structs import SummaryResult

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
        else:
            frozen = [hashmsg(item)]
        item_set.add(frozenset(frozen))
    return frozenset(item_set)


def print_summary(result: SummaryResult) -> None:
    # top line
    print("Exact Match" if result.exact_match else "Not Exact Match")

    # score
    table = [[k, v] for k, v in result.score.graph_kernel_scores.items()]
    print(
        tabulate(
            table,
            headers=["Kernel", "Score"],
            tablefmt="rounded_outline",
            floatfmt=".4f",
        )
    )

    # difference.
    data = []
    for key, matches in result.graph_matches.items():
        data.append(
            [
                f"Graph.{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    for key, matches in result.root_matches.items():
        data.append(
            [
                f"{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    print(
        tabulate(
            data, headers=["Matching Fields", "A", "B"], tablefmt="rounded_outline"
        )
    )


def get_score_embedding(result: SummaryResult) -> np.ndarray:
    return np.array(list(result.score.graph_kernel_scores.values()))


def cos_sim_score(result_a: SummaryResult, result_b: SummaryResult) -> float:
    score_a = get_score_embedding(result_a)
    score_b = get_score_embedding(result_b)
    norm_a = np.linalg.norm(score_a)
    norm_b = np.linalg.norm(score_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(score_a, score_b) / (norm_a * norm_b)


def ang_sim_score(result_a: SummaryResult, result_b: SummaryResult) -> float:
    score = cos_sim_score(result_a, result_b)
    score = np.clip(score, -1.0, 1.0)
    return 1 - (np.arccos(score) / np.pi)
