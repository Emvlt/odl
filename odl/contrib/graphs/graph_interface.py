import importlib.util
from typing import Callable, Dict

from odl.applications.tomo import Geometry

INITIALIZE = False
_exporters: Dict[str, Callable] = {}


def _initialize_if_needed():
    global INITIALIZE
    if not INITIALIZE:
        pygeom_module = importlib.util.find_spec("torch_geometric")
        if pygeom_module is not None:
            from odl.contrib.graphs.backends.pygeom import exporter

            _exporters["torch_geometric"] = exporter
        INITIALIZE = True


def create_graph_from_geometry(geometry: Geometry, scheme: str, backend: str):

    _initialize_if_needed()
    # Gemetry sanity check
    assert isinstance(
        geometry, Geometry
    ), f"The geometry to create the graph from can only be an odl.tomo.Geometry, got {type(geometry)}"

    # scheme sanity check
    assert isinstance(
        scheme, str
    ), f"The scheme to create the graph can only be a str, got {type(scheme)}"
    exporter = _exporters.get(backend)
    if exporter is None:
        raise ValueError(
            f"❌ No exporter found for backend {backend}. Only {list(_exporters.keys())} are registered backends."
        )
    return exporter(geometry, scheme)
