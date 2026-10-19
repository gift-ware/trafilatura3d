from trafilatura3d.nodes.base import Input, Node, Output


class Dot(Node):
    """Dot node (a graph item): reroutes a connection; it costs nothing, the compiler passes its
    input straight through."""

    type = "dot"
    label = "Dot"
    category = "Graph items"
    inputs = {"input": Input(label="")}
    outputs = {"output": Output(label="")}
    passthrough = True
