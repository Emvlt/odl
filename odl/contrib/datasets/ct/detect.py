import odl
import numpy as np

N_PIXELS_DET = 956
SOD = 431.019989
SDD = 529.000488
PIXEL_SIZE = 0.1496

def detect_volume(n_voxels : int, impl:str, device:str):
    return odl.uniform_discr(
        min_pt = [-512,-512],
        max_pt = [ 512, 512],
        shape  = [ n_voxels, n_voxels],
        dtype  = 'float32',
        impl   = impl,
        device = device
    )

def detect_geometry(n_voxels : int):
    det_width = PIXEL_SIZE * N_PIXELS_DET
    FOV_width = det_width * SOD/SDD
    voxel_size =  FOV_width / n_voxels
    scale_factor = 1.0 / voxel_size
    scaled_SOD = SOD * scale_factor
    scaled_SDD = SDD * scale_factor

    scaled_pixel_size = PIXEL_SIZE * scale_factor

    angle_partition = odl.uniform_partition(-1.5*np.pi, 0.5*np.pi , 3600)
    detector_partition = odl.uniform_partition(
        -scaled_pixel_size * N_PIXELS_DET / 2.0, 
        scaled_pixel_size * N_PIXELS_DET / 2.0, 
        N_PIXELS_DET)

    return odl.applications.tomo.FanBeamGeometry(
        angle_partition, detector_partition, src_radius=scaled_SOD, det_radius=scaled_SDD-scaled_SOD)

def detect_ray_trafo(n_voxels = 1024, impl ='pytorch', device='cuda:0'):
    return odl.tomo.RayTransform(
        detect_volume(n_voxels, impl, device), detect_geometry(n_voxels)
        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    ray_trafo = detect_ray_trafo(impl='pytorch', device='cuda:0')

    fbp = odl.applications.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

    sinogram = torch.load('/home/emilien/datasets/processed/2detect/slice00001/mode2/pre_processed_sinogram.pt').squeeze(0)

    tgt = np.load('/home/emilien/datasets/processed/2detect/slice00001/mode2/reconstruction.npy')

    sinogram = ray_trafo.range.element(sinogram)

    rec = fbp(sinogram).data.detach().cpu().numpy()

    plt.matshow(rec)
    plt.savefig('rec')
    plt.clf()

    plt.matshow(tgt)
    plt.savefig('tgt')
    plt.clf()

    plt.matshow(tgt-rec)
    plt.savefig('diff')
    plt.clf()