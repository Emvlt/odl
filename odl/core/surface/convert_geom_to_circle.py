import numpy as np

import odl
from odl.applications.tomo.geometry import ConeBeamGeometry
from surfaces import Cylinder, PointCloud, Circle
from coordinate_systems import CartesianSystem

# Reconstruction space: discretized functions on the cube
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 10)
# Detector: uniformly sampled, n = 512, min = -60, max = 60
detector_partition = odl.uniform_partition(-60, 60, 512)
# Geometry with large fan angle
geometry = odl.applications.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40)

source_positions : np.ndarray= geometry.src_position(geometry.angles)
source_positions = np.column_stack((source_positions, np.full(len(source_positions), 0)))

origin = np.array(list(geometry.translation) + [0])
axis   = np.array([0,0,1])
cart_system = CartesianSystem(origin)

circle = Circle(
        base_center=origin,
        axis = axis,
        radius=geometry.src_radius,
        name="circle"
    )

points = PointCloud(source_positions, system=cart_system, surface=circle)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points.coords[:,0], points.coords[:,1], points.coords[:,2])
plt.savefig('test')

points = points.to_system(circle.coord_system)

assert all([circle.contains_point(p) for p in points])

distances = [points[i].distance_to(points[i+1%len(points)], mode='euclidian') for i in range(len(points)-1)]

print(distances)

distances = [points[i].distance_to(points[i+1%len(points)], mode='surface') for i in range(len(points)-1)]

print(distances)

# sigma = np.mean(distances)

# def kernel(distance, sigma):
#     return np.exp(-distance**2/sigma)

# weights = [kernel(d, sigma) for d in distances]

# print(weights)