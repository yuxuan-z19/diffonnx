from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Matches:
    # Number of items that are the same.
    same: int
    # Total number of items in A.
    a_total: int
    # Total number of items in B.
    b_total: int
    # Items that are in A but not in B.
    a_diff: List[Union[str, frozenset[str]]]
    # Items that are in B but not in A.
    b_diff: List[Union[str, frozenset[str]]]


@dataclass
class Accuracy:
    # Shape and dtype of the output.
    shape: str
    dtype: str

    # Cosine similarity of the output arrays.
    cos_sim: float

    # Maximum absolute error between the output arrays.
    max_err: float


TensorTypeShape = Dict[str, List[int]]


@dataclass
class Profile:
    # Operator label (according to its occurrence in the model).
    inst_label: str

    # Input and output types and shapes
    input_type_shape: List[TensorTypeShape]
    output_type_shape: List[TensorTypeShape]

    # Operator "real" name in the model
    op_name0: str
    # Execution duration in microseconds
    dur0: int
    # torch.FX IR
    ir0: str

    # Optional second operator name (for comparison)
    op_name1: Optional[str] = None
    dur1: Optional[int] = None
    ir1: Optional[str] = None


@dataclass
class StaticResult:
    # Is the structure identical?
    exact_match: bool

    # Similarity score (0 to 1, higher = happier)
    score: Dict[str, float]

    # Detailed matching info
    graph_matches: Dict[str, Matches]

    # Model-level attribute differences
    root_matches: Dict[str, Matches]


OutputPair = Tuple[np.ndarray, np.ndarray]


@dataclass
class ExecutionStats:
    exact_match: bool
    in_invalid: Dict[str, OutputPair]
    out_equal: Dict[str, OutputPair]
    out_nonequal: Dict[str, OutputPair]
    out_mismatched: Dict[str, OutputPair]
    profile_a: List[Profile]
    profile_b: List[Profile]


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

    # Profile comparison among operators.
    profiles: Optional[List[Profile]] = None
