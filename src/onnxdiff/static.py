from onnx import ModelProto, GraphProto

from grakel import Graph
from grakel.kernels import (
    Kernel,
    GraphletSampling,
    Propagation,
    WeisfeilerLehman,
    SubgraphMatching,
)

from .base import Diff
from .utils import hashitem, hashmsg, print_static_summary
from .structs import *

from google.protobuf.message import Message
from typing import List


class GraphDiff:
    KERNELS = [
        WeisfeilerLehman,
        GraphletSampling,
        SubgraphMatching,
        Propagation,
    ]

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def score(self, a_graph: Graph, b_graph: Graph) -> Dict[str, float]:
        graph_kernel_scores = {}
        for kernel_class in self.KERNELS:
            kernel: Kernel = kernel_class(normalize=True, verbose=self._verbose)
            kernel.fit_transform([a_graph])
            score = kernel.transform([b_graph])[0][0]
            graph_kernel_scores[kernel_class.__name__] = score
        return graph_kernel_scores


class StaticDiff(Diff):
    def __init__(
        self,
        model_a: ModelProto,
        model_b: ModelProto,
        verbose: bool = False,
        is_simplified: bool = False,
    ):
        super().__init__(
            model_a=model_a,
            model_b=model_b,
            verbose=verbose,
            is_simplified=is_simplified,
        )

        self.graphdiff = GraphDiff(verbose=verbose)
        self.ngrakel = len(self.graphdiff.KERNELS)

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
                elif input not in initializer_names:
                    print(f"⚠️ No corresponding output found for {input}")

        for i, output in enumerate(graph.output, output_offset):
            if output.name in output_node_hash:
                for src in output_node_hash[output.name]:
                    edge = (src, i + output_offset)
                    edge_list.append(edge)
                    edge_labels[edge] = output.name

        return Graph(edge_list, node_labels=node_labels, edge_labels=edge_labels)

    def _calculate_score(self) -> Score:
        a_graph = self._onnx_to_grakel_graph(self._model_a.graph)
        b_graph = self._onnx_to_grakel_graph(self._model_b.graph)
        graph_kernel_scores = self.graphdiff.score(a_graph, b_graph)

        return Score(graph_kernel_scores=graph_kernel_scores)

    def _match_items(self, a: List[Message], b: List[Message]) -> Matches:
        a_items = hashitem(a)
        b_items = hashitem(b)
        matched = a_items & b_items
        return Matches(
            same=len(matched),
            a_total=len(a_items),
            b_total=len(b_items),
            a_diff=a_items - matched,
            b_diff=b_items - matched,
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
        a_graph = self._model_a.graph
        b_graph = self._model_b.graph
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
                self._model_a,
                self._model_b,
                ignore_fields=["graph"],
            ),
        }

    def summary(self, output=False) -> StaticResult:
        result = StaticResult(
            exact_match=(self._model_a == self._model_b),
            score=self._calculate_score(),
            graph_matches=self._calculate_graph_matches(),
            root_matches=self._calculate_root_matches(),
        )

        if output:
            print_static_summary(result)

        return result
