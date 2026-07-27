from typing import Any, Dict, Optional

import networkx

from odl.applications.tomo import Geometry
from odl.contrib.graphs.backends.base import compute_graph_attribute

def exporter(
    geometry: Geometry,
    scheme: str,
    edges_kwargs: Optional[Dict[str, Any]] = None,
    weights_kwargs: Optional[Dict[str, Any]] = None,
):
    """Geometry Exporter to NetworkX Geometric

    Args:
        geometry (Geometry): ODL geometry object
        scheme (str): scheme used to compute the graph attributes
        edges_kwargs: extra keyword arguments forwarded to the edges calculator (eg. connectivity)
        weights_kwargs: extra keyword arguments forwarded to the weights calculator (eg. sigma, kernel_type, distance_mode)

    Returns:
        Data: PyGeom Data object
    """
    edges = compute_graph_attribute("edges", geometry, scheme, **(edges_kwargs or {}))
    weights = compute_graph_attribute("weights", geometry, scheme, **(weights_kwargs or {}))

    G = networkx.Graph()

    for edge, weight in zip(edges, weights):
        G.add_edge(edge[0], edge[1], weight=weight)
    
    return G
