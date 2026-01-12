import numpy as np

import odl
from odl.applications.tomo.geometry import ConeBeamGeometry
from surfaces import Cylinder, PointCloud
from coordinate_systems import CartesianSystem

# Reconstruction space: discretized functions on the cube
# [-20, 20]^2 x [0, 40] with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300],
    dtype='float32')

# Make a helical cone beam geometry with flat detector
# Angles: uniformly spaced, n = 2000, min = 0, max = 8 * 2 * pi
angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 10)
# Detector: uniformly sampled, n = (512, 64), min = (-50, -3), max = (50, 3)
detector_partition = odl.uniform_partition([-50, -3], [50, 3], [512, 64])
# Spiral has a pitch of 5, we run 8 rounds (due to max angle = 8 * 2 * pi)
geometry = odl.applications.tomo.ConeBeamGeometry(
    angle_partition, detector_partition, src_radius=150, det_radius=100,
    pitch=5.0)

source_positions = geometry.src_position(geometry.angles)

cart_system = CartesianSystem(geometry.translation)

cylinder = Cylinder(
        base_center=geometry.translation,
        axis=geometry.axis,
        radius=geometry.src_radius,
        name="cylinder"
    )

points = PointCloud(source_positions, system=cart_system, surface=cylinder)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points.coords[:,0], points.coords[:,1], points.coords[:,2])
plt.savefig('test')

points = points.to_system(cylinder.coord_system)

assert all([cylinder.contains_point(p) for p in points])

distances = [points[i].distance_to(points[i+1%len(points)], mode='euclidian') for i in range(len(points)-1)]

print(distances)

distances = [points[i].distance_to(points[i+1%len(points)], mode='surface') for i in range(len(points)-1)]

print(distances)

# sigma = np.mean(distances)

# def kernel(distance, sigma):
#     return np.exp(-distance**2/sigma)

# weights = [kernel(d, sigma) for d in distances]

# print(weights)