from typing import Dict, Iterable, List, Optional, Type

from google.protobuf.message import Message
from grakel import Graph
from grakel.kernels import (
    GraphletSampling,
    Kernel,
    Propagation,
    SubgraphMatching,
    WeisfeilerLehman,
)
from onnx import GraphProto, ModelProto

from .base import Diff
from .structs import *
from .utils import hashitem, hashmsg, print_static_summary

KernelLike = Kernel | Iterable[Kernel]


class GraphDiff:
    DEFAULT_KERNEL_CLASSES: List[Type[Kernel]] = [
        WeisfeilerLehman,
        GraphletSampling,
        SubgraphMatching,
        Propagation,
    ]

    def __init__(
        self,
        kernels: Optional[Iterable[Kernel]] = None,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self.kernels: Dict[str, Kernel] = (
            {self._get_name(k): k for k in kernels}
            if kernels is not None
            else self._make_default_kernels()
        )

    def _make_default_kernels(self) -> Dict[str, Kernel]:
        return {
            cls.__name__: cls(normalize=True) for cls in self.DEFAULT_KERNEL_CLASSES
        }

    def _get_name(self, kernel: Kernel) -> str:
        if not isinstance(kernel, Kernel):
            raise TypeError(f"Expected Kernel, got {type(kernel)}")
        return kernel.__class__.__name__

    def add_kernels(self, kernels: KernelLike) -> None:
        if not isinstance(kernels, Iterable):
            kernels = [kernels]

        for kernel in kernels:
            name = self._get_name(kernel)
            operation = "Replaced" if name in self.kernels else "Added"
            self.kernels[name] = kernel
            if self._verbose:
                print(f"<GraphDiff> {operation} kernel: {name}")

    def remove_kernels(self, kernels: KernelLike):
        if not isinstance(kernels, Iterable):
            kernels = [kernels]

        for kernel in kernels:
            name = self._get_name(kernel)
            if name in self.kernels:
                del self.kernels[name]
                if self._verbose:
                    print(f"<GraphDiff> Removed kernel: {name}")
            else:
                if self._verbose:
                    print(f"<GraphDiff> Kernel {name} not found, skipping.")

    def score(self, a_graph: Graph, b_graph: Graph) -> Dict[str, float]:
        graph_kernel_scores: Dict[str, float] = {}

        for name, kernel in self.kernels.items():
            try:
                kernel.fit_transform([a_graph])
                score = kernel.transform([b_graph])[0][0]
                graph_kernel_scores[name] = score
                if self._verbose:
                    print(f"<GraphDiff> Kernel {name}: score = {score:.4f}")
            except Exception as e:
                raise RuntimeError(
                    f"<GraphDiff> Kernel {name} failed during scoring: {e}"
                )

        return graph_kernel_scores

    def __len__(self) -> int:
        return len(self.kernels)


class StaticDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        graphdiff: Optional[GraphDiff] = None,
        verbose: bool = False,
    ):
        super().__init__(
            model_a=model_a,
            model_b=model_b,
            verbose=verbose,
        )

        self.graphdiff = graphdiff or GraphDiff(verbose=verbose)

    def _onnx_to_grakel_graph(self, graph: GraphProto) -> Graph:
        edge_list = []
        node_labels = {}
        edge_labels = {}

        nodes = graph.node
        initializer_names = [init.name for init in graph.initializer]
        output_node_hash = {}

        for i, node in enumerate(nodes, 0):
            node_labels[i] = hashmsg(node)

            for output in node.output:
                output_node_hash.setdefault(output, []).append(i)

        input_offset = len(nodes)
        for i, inp in enumerate(graph.input, input_offset):
            node_labels[i] = hashmsg(inp)
            output_node_hash[inp.name] = [i]

        output_offset = input_offset + len(graph.input) + 1
        for i, out in enumerate(graph.output, output_offset):
            node_labels[i + output_offset] = hashmsg(out)

        for i, node in enumerate(nodes, 0):
            for input in node.input:
                if input in output_node_hash:
                    for src in output_node_hash[input]:
                        edge = (src, i)
                        edge_list.append(edge)
                        edge_labels[edge] = input
                elif (input not in initializer_names) and self._verbose:
                    print(f"<StaticDiff> ⚠️ No corresponding output found for {input}")

        for i, output in enumerate(graph.output, output_offset):
            if output.name in output_node_hash:
                for src in output_node_hash[output.name]:
                    edge = (src, i + output_offset)
                    edge_list.append(edge)
                    edge_labels[edge] = output.name

        return Graph(edge_list, node_labels=node_labels, edge_labels=edge_labels)

    def _calculate_score(self) -> Dict[str, float]:
        a_graph = self._onnx_to_grakel_graph(self.model_a.graph)
        b_graph = self._onnx_to_grakel_graph(self.model_b.graph)
        graph_score = self.graphdiff.score(a_graph, b_graph)
        return graph_score

    def _match_items(self, a: List[Message], b: List[Message]) -> Matches:
        a_items = hashitem(a)
        b_items = hashitem(b)
        matched = a_items & b_items
        return Matches(
            same=len(matched),
            a_total=len(a_items),
            b_total=len(b_items),
            a_diff=sorted(a_items - matched),
            b_diff=sorted(b_items - matched),
        )

    def _get_items_from_fields(self, root: GraphProto, ignore_fields=[]):
        items = []
        for field in root.DESCRIPTOR.fields:
            if field.name not in ignore_fields:
                item = getattr(root, field.name)
                items.append(item)
        return items

    def _match_fields(self, a, b, ignore_fields=[]):
        a_items = self._get_items_from_fields(root=a, ignore_fields=ignore_fields)
        b_items = self._get_items_from_fields(root=b, ignore_fields=ignore_fields)
        return self._match_items(a=a_items, b=b_items)

    def _calculate_graph_matches(self) -> Dict[str, Matches]:
        a_graph = self.model_a.graph
        b_graph = self.model_b.graph
        return {
            "initializers": self._match_items(a_graph.initializer, b_graph.initializer),
            "inputs": self._match_items(a_graph.input, b_graph.input),
            "outputs": self._match_items(a_graph.output, b_graph.output),
            "nodes": self._match_items(a_graph.node, b_graph.node),
            "misc": self._match_fields(
                a_graph,
                b_graph,
                ignore_fields=["initializer", "input", "output", "node"],
            ),
        }

    def _calculate_root_matches(self) -> Dict[str, Matches]:
        return {
            "misc": self._match_fields(
                self.model_a,
                self.model_b,
                ignore_fields=["graph"],
            ),
        }

    def summary(self, output=False) -> StaticResult:
        result = StaticResult(
            exact_match=(self.model_a == self.model_b),
            score=self._calculate_score(),
            graph_matches=self._calculate_graph_matches(),
            root_matches=self._calculate_root_matches(),
        )

        if output:
            print_static_summary(result)

        return result
