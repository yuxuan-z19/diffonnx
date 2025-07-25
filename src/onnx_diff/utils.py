from enum import Enum
from tabulate import tabulate
from colorama import init as colorama_init
from colorama import Fore

from onnx_diff.structs import SummaryResults

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


def print_summary(results: SummaryResults) -> None:
    # top line
    text = (
        "Exact Match"
        if results.exact_match and results.score == 1.0
        else "Difference Detected"
    )
    print("")
    print(f" {text} ({round(results.score * 100, 6)}%)")
    print("")

    # table.
    data = []
    for key, matches in results.graph_matches.items():
        data.append(
            [
                f"Graph.{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    for key, matches in results.root_matches.items():
        data.append(
            [
                f"{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    print(tabulate(data, headers=["Matching Fields", "A", "B"], tablefmt="rounded_outline"))
