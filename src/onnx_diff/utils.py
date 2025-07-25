from enum import Enum
from tabulate import tabulate
from colorama import init as colorama_init
from colorama import Fore
import numpy as np

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
            floatfmt=".3f",
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
