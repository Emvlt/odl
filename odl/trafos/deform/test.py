import numpy as np
import torch
from torch.nn.functional import grid_sample
from scipy.ndimage import map_coordinates
from pykeops.torch import LazyTensor
import matplotlib.pyplot as plt

# The idea would be to design a non linear deformation operator that has a fixed template I0

# then we should develop an ODE solver operator such that given a velocity field v, we could
# compute ode_solver(v) \in I0.space.tangent_bundle, 
# and NonLinearDeformFixedTempl: I0.space.tangent_bundle -> I0.space

def deformation(img, phi, impl='pytorch', interp='linear', boundary='zeros'):
    """
    img: image you want to deform. shape: 2d: HxW; In 3d would be HxWxD
    phi: diffeomorphism to deform the image. shape: 2d: HxWx2;  In 3d would be HxWxDx3
    impl: backend (numpy or pytorch for the moment)
    interp: interpolation method, options: {linear, nearest, cubic}
    boundary: how to deal with the boundaries, options: {zeros, nearest, reflection}
    """
    H, W = img.shape

    dx = phi[..., 0]
    dy = phi[..., 1]

    if impl == 'pytorch':
        dic_interp = {'linear': 'bilinear', 'nearest': 'nearest', 'cubic': 'bicubic'}
        dic_border = {'zeros': 'zeros', 'nearest':'border', 'reflection': 'reflection'}
        interp_torch = dic_interp[interp]
        border_torch = dic_border[boundary]
        imgh = img.unsqueeze(0).unsqueeze(0)
        coords = torch.stack((dy, dx), dim=-1).unsqueeze(0)
        warped = grid_sample(imgh, coords, mode=interp_torch, align_corners=True, padding_mode=border_torch)[0, 0]
        return warped # H x W
    
    elif impl == 'numpy':
        dic_interp = {'linear': 1, 'nearest': 0, 'cubic': 3}
        dic_border = {'zeros': 'constant', 'nearest':'nearest', 'reflection': 'reflect'}
        interp_np = dic_interp[interp]
        border_np = dic_border[boundary]
        coords = np.vstack([ dy.ravel(), dx.ravel() ])
        warped = map_coordinates(img, coords, order=interp_np, mode=border_np).reshape(H, W)
        return warped # H x W
    else:
        pass

def integrate_flow(v_func, impl='pytorch',interp='linear', boundary='zeros'):
    """
        v_t: n_steps x H x W x 2
    """
    dic_module = {'pytorch':torch, 'numpy': np}
    xp = dic_module[impl]

    n_steps, H, W = v_func.shape[:-1]

    ys = xp.linspace(-1, 1, H)
    xs = xp.linspace(-1, 1, W)

    Y, X = xp.meshgrid(ys, xs, indexing='ij')

    phi = xp.stack((Y, X), dim=-1)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        v_ti = v_func[i] 
        v_x = v_ti[...,0]    # H×W
        v_y = v_ti[...,1]    # H×W

        v_x_warp = deformation(v_x, phi, impl, interp, boundary)
        v_y_warp = deformation(v_y, phi, impl, interp, boundary)

        v_phi = xp.stack( (v_y_warp, v_x_warp), axis = -1 )
        phi = phi + dt * v_phi
    return phi # H x W x 2

def dataloss(I0, I1, impl):
    def __dataloss(v_t):
        warped = deformation(I0, integrate_flow(v_t, impl))
        dic_module = {'pytorch': torch, 'numpy': np}
        xp = dic_module[impl]
        return xp.sum(xp.abs(warped - I1)**2)
    return __dataloss

def regloss(grid, sigma=0.1, impl='pykeops'):
    """
    Regularization term. sum_0<=t<=1  v(., t)^T K(., .) v(., t)dt

    Kernel could be a users choice in principle but for the moment -
    - keeping it simple with the gaussian kernel which is the most common -
    - in the literature. The idea is that the kernel quadratic form should -
    - be on a sparse or downsampled grid for eficiency purposes.

    grid: downsample sparse grid, shape H' x W' x 2
    sigma: gaussian variance parameter
    impl: Kernel structure

    """
    if impl=='pykeops':
        def __regloss(v_t):
            n_steps = v_t.shape[0]
            coords = grid.view(-1, 2) # (H'*W', 2)
            x_i = LazyTensor(coords[:, None, :]) 
            x_j = LazyTensor(coords[None, :, :]) 
            sqdist = ((x_i - x_j) ** 2).sum(-1)
            Kij = (- sqdist / (2 * sigma**2)).exp()  
            total = torch.tensor(0.0, device=v_t.device)
            for t in range(n_steps):
                v_j = LazyTensor(v_t[t].view(-1, 2)[None, :, :]) 
                K_v = (Kij * v_j).sum(dim=1)            
                K_v = K_v.view(-1, 2)         
                total += torch.sum((K_v ** 2).sum(dim=1))   
            return total / n_steps
    return __regloss



def main():
    H, W = 64, 64
    x = torch.linspace(-1, 1, H)
    y = torch.linspace(-1, 1, W)
    Y, X = torch.meshgrid(y, x, indexing='ij')  
    grid = torch.stack((Y, X), dim=-1)


    I0 = torch.where(grid[:, :, 0]**2/0.65 + grid[:, :, 1]**2/0.8 <=1, 1.0, 0.0)



    epsilon = 0.15

    phi_x = X + epsilon * torch.sin(torch.pi * Y)
    phi_y = Y + epsilon * torch.sin(torch.pi * X)
    phi = torch.stack((phi_y, phi_x), dim=-1)

    


    deformed = deformation(I0, phi,
                        impl='pytorch',
                        interp='linear',
                        boundary='zeros')

    


    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.title("Original ellipse"); plt.imshow(I0, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title("phi_x, phi_y"); plt.quiver(X[::8,::8], Y[::8,::8],
                                                            (phi_x-X)[::8,::8],
                                                            (phi_y-Y)[::8,::8])
    plt.axis('equal'); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Deformed"); plt.imshow(deformed.detach(), cmap='gray'); plt.axis('off')
    plt.tight_layout()
    plt.show()

    