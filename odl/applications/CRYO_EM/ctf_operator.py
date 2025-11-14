import torch
from torch import nn, fft
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType
import math

# OPTIMISE BY PROJECTING OUT THE Z AXIS AT THE START

import odl
from odl.core.operator import Operator
from odl.core.discr import DiscretizedSpace
from odl.trafos import FourierTransform
from odl.core.phantom import shepp_logan

class CTF(nn.Module):
    def __init__(
        self,
        pixel_width: float,
        sidelength: int,
        ctf_pad: int,
        spherical_aberration: float,
        amplitude_contrast_ratio: float,
        defocus: float,
        electron_energy: float,
    ):
        super().__init__()

        self.register_buffer(
            "ctf_pad",
            torch.tensor(ctf_pad),
        )
        self.register_buffer(
            "ACR",
            torch.tensor(amplitude_contrast_ratio),
        )
        self.register_buffer(
            "defocus",
            torch.tensor(defocus),
        )
        self.register_buffer(
            "aberr",
            torch.tensor(spherical_aberration * 1e7),
        )  # mm -> Ångström
        E = electron_energy * 1e3
        self.register_buffer(
            "wavelen",
            torch.tensor(12.2639 / np.sqrt(E + 0.97845e-6 * E**2)),
        )  # (keV -> eV, m -> Ångström), https://www.jeol.com/words/emterms/20121023.071258.php
        self.register_buffer(
            "kernel",
            self.create_kernel(pixel_width, sidelength, ctf_pad),
        )

    def compute_ctf(self, normsqr):
        term = 0.25 * self.aberr * (self.wavelen**3) * (normsqr**2)
        angle = (0.5 * self.defocus * self.wavelen * normsqr - term) * 2.0 * np.pi
        out = -1.0 * (
            (1.0 - self.ACR**2).sqrt() * torch.sin(angle) + self.ACR * torch.cos(angle)
        )
        return out

    def create_kernel(
        self,
        pixel_width: float,
        sidelength: int,
        ctf_pad: int,
    ) -> TensorType["R_CTF", "R_CTF"]:
        its = fft.fftfreq(sidelength + 2 * ctf_pad, d=pixel_width)
        x = its.unsqueeze(1)
        y = its.unsqueeze(0)
        normsqr = x**2 + y**2
        return self.compute_ctf(normsqr)

    def forward(self, projs: TensorType[..., "R", "R"]) -> TensorType[..., "R", "R"]:
        projs_pad = F.pad(projs, [self.ctf_pad] * 4)
        projs_fourier = fft.fft2(projs_pad) * self.kernel
        projs_ctf = fft.ifft2(projs_fourier)
        if self.ctf_pad > 0:
            return projs_ctf[
                ..., self.ctf_pad : -self.ctf_pad, self.ctf_pad : -self.ctf_pad
            ].real
        else:
            return projs_ctf.real


class CTFOperator(Operator):
    """_summary_

    Args:
        Operator (_type_): _description_
    """

    def __init__(
        self,
        space: DiscretizedSpace,
        ctf_pad: int,
        spherical_aberration: float,
        amplitude_contrast_ratio: float,
        defocus: float,
        electron_energy: float,
        # TODO: There must be a more generic way to describe the change of units, provide a dictionnary mapping str unit to float scaling
        # TODO: Add Documentation to class
        spherical_aberration_scaling=1e7,
        electron_energy_scaling=1e3,
    ):
        # TODO: Add checks on parameters boundaries

        # Extracting the CTF attributes
        self.ctf_pad = ctf_pad
        self.SA = spherical_aberration * spherical_aberration_scaling
        self.ACR = amplitude_contrast_ratio
        self.defocus = defocus
        self.wavelength = 12.2639 / np.sqrt(
            (electron_energy * electron_energy_scaling)
            + 0.97845e-6 * (electron_energy * electron_energy_scaling) ** 2
        )

        self.namespace = space.array_namespace
        self.fourier  = FourierTransform(domain=space)

        # Doing the super init here to access space attribute in _create_kernel
        # Note that the range is the real_space of the input space, otherwise the output is always wrapped inside a complex space
        super(CTFOperator, self).__init__(domain=space, range=space.real_space)

        self.kernel   = self._create_kernel()

        """
        Ideally, we would like to use 
        projs_ctf = self.operator(projs)
        
        self.kernel   = self.fourier.range.element(
            self._create_kernel()
        )        
        self.operator = self.fourier.inverse @ self.kernel @ self.fourier

        But it turns out that the numerical results are inconsistent between ODL's FFT and PyTorch's :(
        """
    def _create_kernel(self):
        # Q (EV) : Do you assume uniform volumes with same shape across all dimensions?
        its = self.namespace.fft.fftfreq(self.domain.shape[0] + 2 * ctf_pad, d=self.domain.cell_volume)
        # Array API friendly unsqueeze
        normsqr = its[:, None] ** 2 + its[None, :] ** 2
        return self._compute_ctf(normsqr)

    def _compute_ctf(self, norm_square):
        term = 0.25 * self.SA * (self.wavelength**3) * (norm_square**2)
        angle = (0.5 * self.defocus * self.wavelength * norm_square - term) * 2 * np.pi
        return -1.0 * (
            math.sqrt(1.0 - self.ACR**2) * self.namespace.sin(angle)
            + self.ACR * self.namespace.cos(angle)
        )

    def _call(self, projs):
        if self.ctf_pad !=0:
            projs = F.pad(projs, [self.ctf_pad] * 4)

        """
        This is an adhoc fix with the namespace. Ideally, we would like to use 
        projs_ctf = self.operator(projs)
        But it turns out that the numerical results are inconsistent between ODL's FFT and PyTorch's :(
        """
        projs_fourier = self.namespace.fft.fft2(projs.asarray()) * self.kernel
        projs_ctf = self.namespace.fft.ifft2(projs_fourier)

        if self.ctf_pad != 0:
            return projs_ctf[
                ..., self.ctf_pad : -self.ctf_pad, self.ctf_pad : -self.ctf_pad
            ].real

        return projs_ctf.real


if __name__ == "__main__":
    pixel_width = 1.0
    sidelength = 64
    ctf_pad = 0
    spherical_aberration = 2.7
    amplitude_contrast_ratio = 0.1
    defocus = 20000
    electron_energy = 300.0

    ### Current implementation
    ctf = CTF(
        pixel_width,
        sidelength,
        ctf_pad,
        spherical_aberration,
        amplitude_contrast_ratio,
        defocus,
        electron_energy,
    )

    reconstruction_space = odl.uniform_discr(
        min_pt=[-32, -32],
        max_pt=[32, 32],
        shape=[64, 64],
        dtype="complex64",
        impl="pytorch",
        device="cpu",
    )

    test_operator = CTFOperator(
        reconstruction_space,
        ctf_pad,
        spherical_aberration,
        amplitude_contrast_ratio,
        defocus,
        electron_energy,
    )

    sample = shepp_logan(reconstruction_space)

    result_odl = test_operator(sample)
    
    import matplotlib.pyplot as plt
    plt.matshow(result_odl.asarray())
    plt.savefig('result_odl')
    plt.clf()

    result_pytorch = ctf(sample.asarray())

    plt.matshow(result_pytorch)
    plt.savefig('result_pytorch')
    plt.clf()

    assert odl.odl_all_equal(result_odl.asarray(), result_pytorch)

