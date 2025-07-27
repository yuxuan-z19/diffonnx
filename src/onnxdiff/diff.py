import onnx
import onnx.checker
from onnx import ModelProto, GraphProto
from onnxsim import simplify

from grakel import Graph
from grakel.kernels import (
    Kernel,
    GraphletSampling,
    Propagation,
    WeisfeilerLehman,
    SubgraphMatching,
)

from . import utils
from .structs import *

from typing import Any
from copy import deepcopy


class OnnxDiff:
    KERNELS = [
        WeisfeilerLehman,
        GraphletSampling,
        SubgraphMatching,
        Propagation,
    ]

    def __init__(self, model_a: ModelProto, model_b: ModelProto, verbose: bool = False):
        self.__check(model_a)
        self.__check(model_b)

        self._verbose = verbose
        self._model_a = self._simplify(model_a)
        self._model_b = self._simplify(model_b)

    def __check(self, model):
        if not isinstance(model, ModelProto):
            raise TypeError(
                f"Model must be an instance of onnx.ModelProto, got: {str(type(model))}"
            )
        if not model.HasField("graph") or not isinstance(model.graph, GraphProto):
            raise ValueError(
                f"Model must contain a valid graph field of type onnx.GraphProto, got: {str(type(model.graph))}"
            )

    def _simplify(self, model: ModelProto) -> ModelProto:
        try:
            model_simplified, _ = simplify(model)
            if self._verbose:
                print("✅ ONNX model simplified successfully.")
            return model_simplified
        except Exception as e:
            if self._verbose:
                print(f"⚠️ ONNX simplification failed: {e}")
            return model

    def _onnx_to_grakel_graph(self, graph: GraphProto) -> Graph:
        edge_list = []
        node_labels = {}
        edge_labels = {}

        nodes = graph.node
        initializer_names = [init.name for init in graph.initializer]
        output_node_hash = {}

        for i, node in enumerate(nodes, 0):
            node_labels[i] = node.op_type
            node_labels[i] += " " + node.name if node.name else ""
            node_labels[i] += " " + node.domain if node.domain else ""

            for output in node.output:
                output_node_hash.setdefault(output, []).append(i)

        input_offset = len(nodes)
        for i, inp in enumerate(graph.input, input_offset):
            node_labels[i] = "Input: " + inp.name
            output_node_hash[inp.name] = [i]

        output_offset = input_offset + len(graph.input) + 1
        for i, out in enumerate(graph.output, output_offset):
            node_labels[i + output_offset] = "Output: " + out.name

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

    def _get_graph_scores(self, a_graph: Graph, b_graph: Graph) -> Dict[str, float]:
        graph_kernel_scores = {}
        for kernel_class in self.KERNELS:
            kernel: Kernel = kernel_class(normalize=True, verbose=self._verbose)
            kernel.fit_transform([a_graph])
            score = kernel.transform([b_graph])[0][0]
            graph_kernel_scores[kernel_class.__name__] = score
        return graph_kernel_scores

    def _calculate_score(self) -> Score:
        a_graph = self._onnx_to_grakel_graph(self._model_a.graph)
        b_graph = self._onnx_to_grakel_graph(self._model_b.graph)
        graph_kernel_scores = self._get_graph_scores(a_graph, b_graph)

        return Score(graph_kernel_scores=graph_kernel_scores)

    def _safe_remove(self, items: list[Any], x) -> bool:
        # Prevents error multiple type list. Sometimes there's no equality operator, so would exit early.
        # Otherwise would call .remove(x)
        for index, item in enumerate(items):
            if type(item) == type(x) and item == x:
                del items[index]
                return True
        return False

    def _match_items(self, a, b) -> Matches:
        a_items = deepcopy(a)
        b_items = deepcopy(b)
        a_total = len(a_items)
        b_total = len(b_items)

        match_count = 0
        while len(a_items) > 0:
            a = a_items.pop()
            if self._safe_remove(b_items, a):
                match_count += 1

        return Matches(same=match_count, a_total=a_total, b_total=b_total)

    def _get_items_from_fields(self, root, ignore_fields=[]):
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

    def _validate(self, model: ModelProto) -> bool:
        try:
            onnx.checker.check_model(model)
            return True
        except:
            return False

    def summary(self, output=False) -> SummaryResult:
        results = SummaryResult(
            exact_match=(self._model_a == self._model_b),
            score=self._calculate_score(),
            a_valid=self._validate(self._model_a),
            b_valid=self._validate(self._model_b),
            graph_matches=self._calculate_graph_matches(),
            root_matches=self._calculate_root_matches(),
        )

        if output:
            utils.print_summary(results)

        return results
