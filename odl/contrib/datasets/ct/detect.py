""" 2Detect dataset binders for ODL (see https://www.nature.com/articles/s41597-023-02484-6)"""

from pathlib import Path
from typing import List 

import numpy as np
import imageio.v2 as imageio
from scipy.interpolate import interp1d

import odl

### Constants
N_PIXELS_DET = 956
SOD = 431.019989
SDD = 529.000488
PIXEL_SIZE = 0.1496
DET_SUBSAMPLING = 2

def detect_volume(n_voxels : int, impl:str, device:str):
    """Create the volume for 2Detect"""
    return odl.uniform_discr(
        min_pt = [-512,-512],
        max_pt = [ 512, 512],
        shape  = [ n_voxels, n_voxels],
        dtype  = 'float32',
        impl   = impl,
        device = device
    )

def detect_geometry(n_voxels : int, angles_indices : List =None):
    """Create the geometry"""
    det_width = PIXEL_SIZE * N_PIXELS_DET
    FOV_width = det_width * SOD/SDD
    voxel_size =  FOV_width / n_voxels
    scale_factor = 1.0 / voxel_size
    scaled_SOD = SOD * scale_factor
    scaled_SDD = SDD * scale_factor

    scaled_pixel_size = PIXEL_SIZE * scale_factor

    angle_partition = odl.uniform_partition(-1.5*np.pi, 0.5*np.pi , 3600)
    if angles_indices is not None:
        angle_partition = angle_partition[angles_indices]
    detector_partition = odl.uniform_partition(
        -scaled_pixel_size * N_PIXELS_DET / 2.0, 
        scaled_pixel_size * N_PIXELS_DET / 2.0, 
        N_PIXELS_DET)

    return odl.applications.tomo.FanBeamGeometry(
        angle_partition, detector_partition, src_radius=scaled_SOD, det_radius=scaled_SDD-scaled_SOD)

def detect_ray_trafo(n_voxels = 1024, impl ='pytorch', device='cuda:0', geometry = None):
    if geometry is None:
        geometry = detect_geometry(n_voxels)
    return odl.applications.tomo.RayTransform(
        detect_volume(n_voxels, impl, device), geometry
        )

def correct_detector_shift(sinogram, slice_index):
        detector_correction = [0,1]

        if (slice_index < 2830) or (5520 < slice_index < 5871):
            detector_shift = detector_correction[0]
        else:
            detector_shift = detector_correction[1]
        ## Apply detector shift
        detector_grid = np.arange(0, 1912)
        # for the sinogram
        detector_grid_shifted = detector_grid + detector_shift
        detector_grid_shift_corrected = interp1d(
            detector_grid,
            sinogram,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )  # type:ignore
        sinogram: np.ndarray = detector_grid_shift_corrected(detector_grid_shifted)
        sinogram = np.ascontiguousarray(sinogram)
        return sinogram


def average_flat_field(flat1, flat2):
    return np.mean(np.array([flat1, flat2]), axis=0)


def apply_subsampling(sinogram, dark, flat):    
    sinogram = (
        (sinogram[:, 0::DET_SUBSAMPLING] + sinogram[:, 1::DET_SUBSAMPLING])
    )[:-1, :]
    dark = dark[0, 0::DET_SUBSAMPLING] + dark[0, 1::DET_SUBSAMPLING]
    flat = flat[0, 0::DET_SUBSAMPLING] + flat[0, 1::DET_SUBSAMPLING]
    return sinogram, dark, flat


def field_corrections(sinogram, dark, flat):
    sinogram = (sinogram - dark) / (flat - dark)
    return sinogram


def apply_log_negative(sinogram):
    sinogram = np.clip(sinogram, min=1e-6)
    sinogram = - np.log(sinogram)
    return sinogram


def preprocess_sinogram(path_to_mode : Path):
    sinogram = imageio.imread(path_to_mode.joinpath("sinogram.tif")).astype("float32")
    flat1 = imageio.imread(path_to_mode.joinpath("flat1.tif")).astype("float32")
    flat2 = imageio.imread(path_to_mode.joinpath("flat2.tif")).astype("float32")
    dark = imageio.imread(path_to_mode.joinpath("dark.tif")).astype("float32")
    # compute average flat field
    flat = average_flat_field(flat1, flat2)
    # correct detector shifts
    [sinogram, dark, flat] = [
        correct_detector_shift(data,1) for data in [sinogram, dark, flat]
        ]
    # Apply detector subsampling
    sinogram, dark, flat = apply_subsampling(sinogram, dark, flat)
    # Apply field corrections
    sinogram = field_corrections(sinogram, dark, flat)
    # Apply log_negative
    sinogram = apply_log_negative(sinogram)

    return sinogram


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ray_trafo = detect_ray_trafo(impl='numpy', device='cpu')

    fbp = odl.applications.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

    mode_path = Path('2detect/slice00001/mode2')

    assert mode_path.is_dir(), f'The specified path {mode_path} is not a directory'

    sinogram = preprocess_sinogram(mode_path)

    reconstruction = np.asarray(imageio.imread(mode_path.joinpath("reconstruction.tif")))

    sinogram = ray_trafo.range.element(sinogram)

    rec = fbp(sinogram).data

    plt.matshow(rec)
    plt.title('Filtered Back-Projection')
    plt.colorbar()
    plt.savefig('FBP')
    plt.clf()

    plt.matshow(reconstruction)
    plt.title('Target Reconstruction')
    plt.colorbar()
    plt.savefig('Target Reconstruction')
    plt.clf()