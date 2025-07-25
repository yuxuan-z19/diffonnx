from dataclasses import dataclass


@dataclass
class Matches:
    same: int  # Number of items that are exactly the same.
    a_total: int  # Total number of items in A.
    b_total: int  # Total number of items in B.


@dataclass
class SummaryResults:
    exact_match: bool  # The entire models are exactly the same.
    score: float  # Graph kernel score to estimate shape similarity.
    a_valid: bool  # True when model A passes ONNX checker.
    b_valid: bool  # True when model B passes ONNX checker.
    graph_matches: dict[
        str, Matches
    ]  # Number of items exactly the same, for all fields in graph.
    root_matches: dict[
        str, Matches
    ]  # Number of items exactly the same, for the fields in root (excluding the graph).