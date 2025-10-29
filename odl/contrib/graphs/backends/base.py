# This module contains code to create a graph object from an ODL geometry
# It is made to be able to export to PyGeom or maybe NetworkX
from functools import wraps
from typing import Any, Callable

import numpy as np

from odl.applications.tomo import Geometry

_calculators = {"node_number": {}, "edges": {}, "weights": {}}

def register_calculator(calculator_type: str, geometry_name: str, scheme: str):
    """Decorator to register a calculator (object that calculates graph attributes from a geometry)

    Args:
        calculator_type (str): name of the calculator
        geometry_name (str): name of the geometry, derived from geometry.__class__.__name__
        scheme (str): scheme used to compute the graph attributes
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        _calculators[calculator_type][(geometry_name, scheme)] = wrapper
        return wrapper

    return decorator


@register_calculator("node_number", "FanBeamGeometry", "GLM")
def node_number_GLM_FanBeam(geometry: Geometry):
    return len(geometry.angles)


@register_calculator("edges", "FanBeamGeometry", "GLM")
def edges_GLM_FanBeam(geometry: Geometry):
    angles = np.sort(geometry.angles)
    edges = []
    n_angles = len(angles)
    for i in range(n_angles):
        edges += [[i, (i + 1) % n_angles], [(i + 1) % n_angles, i]]
    assert len(edges) == 2 * n_angles
    return edges


@register_calculator("weights", "FanBeamGeometry", "GLM")
def weights_GLM_FanBeam(geometry: Geometry):
    angles = np.sort(geometry.angles)
    angular_differences = np.ediff1d(angles, to_begin=angles[-1] - angles[0])
    cosines = np.cos(angular_differences)
    weights = np.column_stack((cosines, cosines)).ravel()
    return weights


def compute_graph_attribute(
    graph_attribute_name: str, geometry: Geometry, scheme: str
) -> np.ndarray | int:
    """Computes a certain graph attribute

    Args:
        graph_attribute_name (str): Name of the graph attribute to compute (node_number, edges, weights)
        geometry (Geometry): name of the geometry, derived from geometry.__class__.__name__
        scheme (str): scheme used to compute the graph attributes

    Raises:
        NotImplementedError: If the chosen graph attribute is not implemented (eg, if it is not in [node_number, edges, weights]) an error will be raised
        NotImplementedError: If the computation for a geometry and a certain weighting scheme is not implemented, an error will be raised

    Returns:
        (np.ndarray | int)
    """
    geometry_name = geometry.__class__.__name__
    calculator_type = _calculators.get(graph_attribute_name)
    if calculator_type is None:
        raise NotImplementedError(
            f"❌ The graph attribute {graph_attribute_name} has no associated way to compute its values"
        )
    calculator = calculator_type.get((geometry_name, scheme))
    if calculator is None:
        raise NotImplementedError(
            f"❌ The calculation of the {graph_attribute_name} for the geometry {geometry_name} and scheme {scheme} is not Implemented"
        )
    return calculator(geometry)
