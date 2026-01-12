import networkx

from odl.applications.tomo import Geometry
from odl.contrib.graphs.backends.base import compute_graph_attribute

def exporter(geometry: Geometry, scheme: str):
    """Geometry Exporter to NetworkX Geometric

    Args:
        geometry (Geometry): ODL geometry object

    Returns:
        Data: PyGeom Data object
    """
    edges = compute_graph_attribute("edges", geometry, scheme)
    weights = compute_graph_attribute("weights", geometry, scheme)

    G = networkx.Graph()

    for edge, weight in zip(edges, weights):
        G.add_edge(edge[0], edge[1], weight=weight)
    
    return G
