import numpy as np
import odl

from graph_interface import create_graph_from_geometry

angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -60, max = 60
detector_partition = odl.uniform_partition(-60, 60, 512)
# Geometry with large fan angle
geometry = odl.applications.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40
)

graph = create_graph_from_geometry(geometry, backend="torch_geometric", scheme="GLM")

print(graph)
