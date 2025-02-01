from engine import Value
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            if (
                v.tree.method != "value"
            ):  # Only add edges for operations, not initial values
                if isinstance(v.tree.operand1, Value):
                    edges.add((v.tree.operand1, v))
                    build(v.tree.operand1)
                if isinstance(v.tree.operand2, Value):
                    edges.add((v.tree.operand2, v))
                    build(v.tree.operand2)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ %s | data %.4f | grad %.4f }"
            % (getattr(n, "label", ""), n.value or 0, n.grad),
            shape="record",
        )
        if n.tree.method != "value":
            dot.node(name=str(id(n)) + n.tree.method, label=n.tree.method)
            dot.edge(str(id(n.tree.operand1)), str(id(n)) + n.tree.method)
            if isinstance(n.tree.operand2, Value):
                dot.edge(str(id(n.tree.operand2)), str(id(n)) + n.tree.method)
            dot.edge(str(id(n)) + n.tree.method, str(id(n)))
    return dot


class kv:
    @staticmethod
    def trace(root):
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    @staticmethod
    def draw_dot(root, format="svg", rankdir="LR"):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ["LR", "TB"]
        nodes, edges = kv.trace(root)
        dot = Digraph(
            format=format, graph_attr={"rankdir": rankdir}
        )  # , node_attr={'rankdir': 'TB'})

        for n in nodes:
            dot.node(
                name=str(id(n)),
                label="{ %s | data %.4f | grad %.4f }"
                % (getattr(n, "label", ""), n.data, n.grad),
                shape="record",
            )
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
