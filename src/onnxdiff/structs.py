from dataclasses import dataclass
from typing import Dict


@dataclass
class Matches:
    same: int  # Number of items that are exactly the same.
    a_total: int  # Total number of items in A.
    b_total: int  # Total number of items in B.


@dataclass
class Score:
    graph_kernel_scores: Dict[str, float]  # Graph kernel scores for each kernel.


@dataclass
class SummaryResult:
    exact_match: bool  # The entire models are exactly the same.
    score: Score  # Graph kernel score to estimate shape similarity.
    a_valid: bool  # True when model A passes ONNX checker.
    b_valid: bool  # True when model B passes ONNX checker.

    # Number of items exactly the same, for all fields in graph.
    graph_matches: Dict[str, Matches]

    # Number of items exactly the same, for the fields in root (excluding the graph).
    root_matches: Dict[str, Matches]
