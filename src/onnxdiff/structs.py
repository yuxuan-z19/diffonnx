from dataclasses import dataclass
from typing import Dict, Set, Union, Tuple


@dataclass
class Matches:
    same: int  # Number of items that are exactly the same.
    a_total: int  # Total number of items in A.
    b_total: int  # Total number of items in B.
    a_diff: Set[Union[str, frozenset[str]]]
    b_diff: Set[Union[str, frozenset[str]]]


@dataclass
class Accuracy:
    shape: str
    dtype: str
    cos_sim: float
    max_err: float


@dataclass
class Score:
    graph_kernel_scores: Dict[str, float]  # Graph kernel scores for each kernel.


@dataclass
class StaticResult:
    exact_match: bool  # The entire models are exactly the same.
    score: Score  # Graph kernel score to estimate shape similarity.

    # Number of items exactly the same, for all fields in graph.
    graph_matches: Dict[str, Matches]

    # Number of items exactly the same, for the fields in root (excluding the graph).
    root_matches: Dict[str, Matches]


@dataclass
class RuntimeResult:
    exact_match: bool
    in_invalid: Dict[str, Accuracy]
    out_equal: Dict[str, Accuracy]
    out_nonequal: Dict[str, Accuracy]
    out_mismatched: Dict[str, Accuracy]
