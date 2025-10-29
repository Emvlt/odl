"""Example using the ray transform with circular cone beam geometry."""

from time import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import odl

RESULTS_FILEPATH = Path('results.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--volume_size', required=True)
parser.add_argument('--n_iterations', required=True)
parser.add_argument('--impl', required=True)
parser.add_argument('--device', required=True)
args = parser.parse_args()

volume_size = int(args.volume_size)
n_iterations = int(args.n_iterations)
impl = args.impl
device = args.device

# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[volume_size, volume_size, volume_size],
    impl=impl,
    device=device,
    dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [512, 512])
geometry = odl.applications.tomo.ConeBeamGeometry(
    angle_partition, detector_partition, src_radius=1000, det_radius=100,
    axis=[1, 0, 0])

# Ray transform (= forward projection).
ray_trafo = odl.applications.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, True)

result_dict = {
    'time' : [],
}

proj_data = ray_trafo(phantom)

for i in range(n_iterations):
    print(f'Processing iteration {i}')
    t0 = time()
    # Create projection data by calling the ray transform on the phantom
    rec_data = ray_trafo.adjoint(proj_data)
    t1 = time()
    result_dict['time'].append(t1-t0)

result_df = pd.DataFrame.from_dict(result_dict)
result_df['impl'] = impl
result_df['volume_size'] = volume_size
result_df['experiment_name'] = 'backward_projection'

if RESULTS_FILEPATH.is_file():
    results_df = pd.read_csv(RESULTS_FILEPATH)
    results_df = pd.concat([results_df, result_df], ignore_index=True, sort=False)

else:
    results_df = result_df

results_df.to_csv(RESULTS_FILEPATH, index=False)


