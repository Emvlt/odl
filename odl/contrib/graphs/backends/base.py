# This module contains code to create a graph object from an ODL geometry
# It is made to be able to export to PyGeom or maybe NetworkX
from functools import wraps
from typing import Any, Callable
import math

import numpy as np

from odl.applications.tomo import Geometry
from odl.core.surface.surfaces import Circle, PointCloud
from odl.core.surface.coordinate_systems import CartesianSystem 

_calculators = {"points": {}, "node_number": {}, "edges": {}, "weights": {}}

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

def gaussian_kernel(distance:float, sigma=1.0):
    return math.exp(-distance**2/sigma)

@register_calculator("points", "FanBeamGeometry", "GLM")
def points_GLM_FanBeam(geometry:Geometry) -> PointCloud:
    # Extract source positions, origin and axis from the geometry object
    source_positions : np.ndarray= geometry.src_position(geometry.angles)
    source_positions = np.column_stack((source_positions, np.full(len(source_positions), 0)))

    origin = np.array(list(geometry.translation) + [0])
    axis   = np.array([0,0,1])
    
    # Extract the Circle surface from the geometry and create the cartesian system in which points initially live
    circle = Circle(
            base_center=origin,
            axis = axis,
            radius=geometry.src_radius,
            name="circle"
        )
    cart_system = CartesianSystem(origin)

    # Attach the PointCloud to the surface and express the points in the right coordinate system
    points = PointCloud(source_positions, system=cart_system, surface=circle)    
    points = points.to_system(circle.coord_system)
    assert all([circle.contains_point(p) for p in points])

    return points


@register_calculator("node_number", "PointCloud", "GLM")
def node_number_GLM_FanBeam(points: PointCloud):
    return len(points)


@register_calculator("edges", "PointCloud", "GLM")
def edges_GLM_FanBeam(points: PointCloud):
    # For the FanBeam 2D geometry, we connect successive measurements 
    edges = []
    n_source_positions = len(points)
    for i in range(n_source_positions):
        edges += [[i, (i + 1) % n_source_positions], [(i + 1) % n_source_positions, i]]
    assert len(edges) == 2 * n_source_positions
    return edges


@register_calculator("weights", "PointCloud", "GLM")
def weights_GLM_FanBeam(points: PointCloud, sigma=1.0):
    distances = [points[i].distance_to(points[(i+1)%len(points)], mode='surface') for i in range(len(points))]
    weights = [gaussian_kernel(distance, sigma) for distance in distances]
    weights = np.column_stack((weights, weights)).ravel()
    return weights


def compute_graph_attribute(
    graph_attribute_name: str, object, scheme: str
) -> np.ndarray | int:
    """Computes a certain graph attribute

    Args:
        graph_attribute_name (str): Name of the graph attribute to compute (node_number, edges, weights)
        object: object to process
        scheme (str): scheme used to compute the graph attributes

    Raises:
        NotImplementedError: If the chosen graph attribute is not implemented (eg, if it is not in [node_number, edges, weights]) an error will be raised
        NotImplementedError: If the computation for a geometry and a certain weighting scheme is not implemented, an error will be raised

    Returns:
        (np.ndarray | int)
    """
    object_name = object.__class__.__name__
    calculator_type = _calculators.get(graph_attribute_name)
    if calculator_type is None:
        raise NotImplementedError(
            f"❌ The graph attribute {graph_attribute_name} has no associated way to compute its values"
        )
    calculator = calculator_type.get((object_name, scheme))
    if calculator is None:
        raise NotImplementedError(
            f"❌ The calculation of the {graph_attribute_name} for the object {object_name} and scheme {scheme} is not Implemented"
        )
    return calculator(object)
