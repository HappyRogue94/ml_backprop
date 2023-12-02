from graphviz import Digraph

def trace(root):
    """
    builds a set of nodes and edges in a graph tree
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    pass