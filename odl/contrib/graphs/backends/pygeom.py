# This module contains code to export an ODL geometry to a PyGeom Data object
from typing import Any, Dict, Optional

from torch_geometric.data import Data
import torch

from odl.applications.tomo import Geometry
from odl.contrib.graphs.backends.base import compute_graph_attribute


def exporter(
    geometry: Geometry,
    scheme: str,
    points_kwargs: Optional[Dict[str, Any]] = None,
    node_number_kwargs: Optional[Dict[str, Any]] = None,
    edges_kwargs: Optional[Dict[str, Any]] = None,
    weights_kwargs: Optional[Dict[str, Any]] = None,
) -> Data:
    """Geometry Exporter to Pytorch Geometric

    Args:
        geometry (Geometry): ODL geometry object
        scheme (str): scheme used to compute the graph attributes
        points_kwargs: extra keyword arguments forwarded to the points calculator
        node_number_kwargs: extra keyword arguments forwarded to the node_number calculator
        edges_kwargs: extra keyword arguments forwarded to the edges calculator (eg. connectivity)
        weights_kwargs: extra keyword arguments forwarded to the weights calculator (eg. sigma, kernel_type, distance_mode)

    Returns:
        Data: PyGeom Data object
    """
    points = compute_graph_attribute("points", geometry, scheme, **(points_kwargs or {}))
    num_nodes = compute_graph_attribute("node_number", points, scheme, **(node_number_kwargs or {}))
    edges = compute_graph_attribute("edges", points, scheme, **(edges_kwargs or {}))
    weights = compute_graph_attribute("weights", points, scheme, **(weights_kwargs or {}))

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    weights = torch.tensor(weights, dtype=torch.float)
    return Data(num_nodes=num_nodes, edge_index=edges, edge_weight=weights)
