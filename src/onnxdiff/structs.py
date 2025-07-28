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
    # Graph kernel scores for each kernel.
    graph_kernel_scores: Dict[str, float]


@dataclass
class StaticResult:
    # Is the structure identical?
    exact_match: bool
    
    # Similarity score (0 to 1, higher = happier)
    score: Score
    
    # Detailed matching info
    graph_matches: Dict[str, Matches] 

    # Model-level attribute differences
    root_matches: Dict[str, Matches]


@dataclass
class RuntimeResult:
    # Are the outputs exactly the same?
    exact_match: bool

    # Inputs that are invalid (shape/dtype mismatch).
    invalid: Dict[str, Accuracy]

    # Outputs that are exactly the same.
    equal: Dict[str, Accuracy]

    # Outputs that are not exactly the same but have the same shape and dtype.
    nonequal: Dict[str, Accuracy]

    # Outputs that have shape/dtype mismatch.
    mismatched: Dict[str, Accuracy]
