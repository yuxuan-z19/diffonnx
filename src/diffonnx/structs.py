from dataclasses import dataclass, field

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
    a_diff: list[str | frozenset[str]]
    # Items that are in B but not in A.
    b_diff: list[str | frozenset[str]]


@dataclass
class Accuracy:
    # Shape and dtype of the output.
    shape: str
    dtype: str

    # Cosine similarity of the output arrays.
    cos_sim: float

    # Maximum absolute error between the output arrays.
    max_err: float


TensorTypeShape = dict[str, list[int]]


@dataclass
class Profile:
    # Operator label (according to its occurrence in the model).
    inst_label: str

    # Input and output types and shapes
    input_type_shape: list[TensorTypeShape]
    output_type_shape: list[TensorTypeShape]

    # Operator "real" name in the model
    op_name0: str
    # Execution duration in microseconds
    dur0: int
    # torch.FX IR
    ir0: str

    # Optional second operator name (for comparison)
    op_name1: str | None = None
    dur1: int | None = None
    ir1: str | None = None


@dataclass
class StaticResult:
    # Is the structure identical?
    exact_match: bool

    # Similarity score (0 to 1, higher = happier)
    score: dict[str, float]

    # Detailed matching info
    graph_matches: dict[str, Matches]

    # Model-level attribute differences
    root_matches: dict[str, Matches]


OutputPair = tuple[np.ndarray, np.ndarray]


@dataclass
class ExecutionStats:
    exact_match: bool
    in_invalid: dict[str, OutputPair]
    out_equal: dict[str, OutputPair]
    out_nonequal: dict[str, OutputPair]
    out_mismatched: dict[str, OutputPair]
    profile_a: list[Profile]
    profile_b: list[Profile]


@dataclass
class RuntimeResult:
    # Are the outputs exactly the same?
    exact_match: bool

    # Inputs that are invalid (shape/dtype mismatch).
    invalid: dict[str, Accuracy]

    # Outputs that are exactly the same.
    equal: dict[str, Accuracy]

    # Outputs that are not exactly the same but have the same shape and dtype.
    nonequal: dict[str, Accuracy]

    # Outputs that have shape/dtype mismatch.
    mismatched: dict[str, Accuracy]

    # Profile comparison among operators.
    profiles: list[Profile] = field(default_factory=list)
