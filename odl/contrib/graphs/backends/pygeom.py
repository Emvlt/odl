# This module contains code to export an ODL geometry to a PyGeom Data object
from torch_geometric.data import Data
import torch

from odl.applications.tomo import Geometry
from odl.contrib.graphs.backends.base import compute_graph_attribute


def exporter(geometry: Geometry, scheme: str) -> Data:
    """Geometry Exporter to Pytorch Geometric

    Args:
        geometry (Geometry): ODL geometry object

    Returns:
        Data: PyGeom Data object
    """
    points = compute_graph_attribute("points", geometry, scheme)
    num_nodes = compute_graph_attribute("node_number", points, scheme)
    edges = compute_graph_attribute("edges", points, scheme)
    weights = compute_graph_attribute("weights", points, scheme)

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    weights = torch.tensor(weights, dtype=torch.float)
    return Data(num_nodes=num_nodes, edge_index=edges, edge_weight=weights)
