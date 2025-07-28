# Diff your ONNXs – Spot the Difference (Like, Seriously)

A **powerful yet playful** tool to **compare and analyze ONNX models** – whether you're hunting for hidden changes or debugging mysterious outputs. Think of it as a microscope for your pair of models, complete with structure analysis and runtime sanity checks.

## 💥 Why ONNXDiff?

So, picture this:

*You’ve got two PyTorch models. Your boss says, "We need formal verification to prove these models are **diverse**, and you’ve got, oh, 3 days."* 😅

You Google, or prompt an LLM, frantically. You find something called [`onnx-diff`](https://pypi.org/project/onnx-diff/). **Perfect!** Except… it’s closed-source.Modifying it? **Not happening.**

So what do you do?
You roll up your sleeves and build your own version. From scratch. With improvements. And a dash of spite.

**Introducing: `onnxdiff` (note the lack of a hyphen `-`).**

## 🎯 What It Does

### 🧠 Static Analysis

- Full ONNX graph comparison (nodes, edges, initializers, etc.)
- Graph kernel similarity scores based on [GraKel](https://github.com/ysig/GraKeL)
- Operator statistics and diff
- Comparison after model simplification

#### Why These Kernels?

We chose the following four kernels because they **offer strong structural sensitivity, good scalability, and semantic insight**, all critical for comparing ONNX models in optimization, verification, and architecture evolution.

| Kernel                | Strengths                                 | Focus                |
| --------------------- | ----------------------------------------- | -------------------- |
| **Weisfeiler-Lehman** | Fast, captures subtle structural changes  | Global structure     |
| **Graphlet**          | Highlights local connection patterns      | Local structure      |
| **Subgraph Matching** | Detects reused or rewritten modules       | Block-level changes  |
| **Propagation**       | Combines topology with operator semantics | Semantic differences |

Together, they provide a **balanced and interpretable toolkit** for robust ONNX model analysis.

### ⚡ Runtime Analysis

- Cosine / angular similarity of tensor outputs
- Maximum absolute error inspection
- Precision breakdowns
- Supports multiple execution providers (CPU, CUDA, anything ONNXRuntime speaks)

## 🚀 Installation

We recommend using `uv` — a faster, modern Python package manager.

```bash
pip install onnxdiff
```

Or if you’re the DIY type:

```bash
git clone https://github.com/yuxuan-z19/onnxdiff.git
cd onnxdiff && pip install -e .
```

## 🛠 Usage

### CLI: For Fast Hands-On Nerding

```bash
# Basic comparison
onnxdiff ref_model.onnx usr_model.onnx

# Verbose mode – unleash the diff dragon
onnxdiff ref_model.onnx usr_model.onnx -v
```

![demo](./assets/demo.png)

### Python API: For People Who Think in Code

```python
import onnx
from onnxdiff import OnnxDiff, StaticDiff, RuntimeDiff

ref = onnx.load("ref_model.onnx")
usr = onnx.load("usr_model.onnx")

# Full analysis, max drama
diff = OnnxDiff(ref, usr, verbose=True)
static_result, runtime_result = diff.summary(output=True)

# Just structure? Sure
static_only = StaticDiff(ref, usr)
static_result = static_only.summary(output=True)

# Runtime only, CUDA powered
runtime_only = RuntimeDiff(
    ref, usr, providers=["CUDAExecutionProvider"]
)
runtime_result = runtime_only.summary(output=True)
```

### Quick Graph Kernel Scores

For users who want a simple, unified interface to compute multiple graph kernel similarities at once, we provide the handy `GraphDiff` class. It bundles the kernels and returns their scores in one call:

```python
from grakel.graph import Graph
from onnxdiff.static import GraphDiff

# graph_a = Graph()
# graph_b = Graph()

graph_diff = GraphDiff(verbose=True)
graph_diff.add_kernels(
    [
        ShortestPath(normalize=True, with_labels=False),
        RandomWalkLabeled(normalize=True),
    ]
)
graph_diff.remove_kernels([WeisfeilerLehman()])

scores = graph_diff.score(graph_a, graph_b)
print(scores.keys())
# Example output:
# {'GraphletSampling', 'SubgraphMatching', 'Propagation', 'ShortestPath', 'RandomWalkLabeled'}
```

This way, you can easily evaluate graph similarity across multiple kernels without manually instantiating each one.

## 📊 Output Breakdown

```python
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
```

```python
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
```

Check `onnxdiff.structs` for more about `Matches` and `Accuracy`.

## 👷 Development

If you're contributing or running tests:

```bash
git clone https://github.com/yuxuan-z19/onnxdiff.git
cd onnxdiff

# Install dev dependencies and lock the environment
uv sync --locked --all-extras --dev

# Run tests
uv run pytest -n auto
```
