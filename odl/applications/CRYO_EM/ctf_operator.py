import torch
from torch import nn, fft
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType
#OPTIMISE BY PROJECTING OUT THE Z AXIS AT THE START

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
            torch.tensor(spherical_aberration*1e7),
        ) # mm -> Ångström
        E = electron_energy * 1e3
        self.register_buffer(
            "wavelen", 
            torch.tensor(12.2639/np.sqrt(E+0.97845e-6*E**2)),
        ) # (keV -> eV, m -> Ångström), https://www.jeol.com/words/emterms/20121023.071258.php
        self.register_buffer(
            "kernel", 
            self.create_kernel(pixel_width, sidelength, ctf_pad),
        )
        
    def compute_ctf(self, normsqr):
        term = 0.25 * self.aberr * (self.wavelen ** 3) * (normsqr ** 2)
        angle = (0.5 * self.defocus * self.wavelen * normsqr - term) * 2. * np.pi
        out = -1. * (
            (1. - self.ACR ** 2).sqrt() * torch.sin(angle) 
            + self.ACR * torch.cos(angle)
        )
        return out
    
    def create_kernel(
            self,
            pixel_width: float,
            sidelength: int,
            ctf_pad: int,
        ) -> TensorType["R_CTF", "R_CTF"]:
        its = fft.fftfreq(sidelength+2*ctf_pad, d=pixel_width)
        x = its.unsqueeze(1)
        y = its.unsqueeze(0)
        normsqr = x ** 2 + y ** 2
        return self.compute_ctf(normsqr)
    
    def forward(
            self, 
            projs: TensorType[..., "R", "R"]
        ) -> TensorType[..., "R", "R"]:
        projs_pad = F.pad(projs, [self.ctf_pad]*4)
        projs_fourier = fft.fft2(projs_pad) * self.kernel
        projs_ctf = fft.ifft2(projs_fourier)
        if self.ctf_pad > 0:
            return projs_ctf[
                ..., 
                self.ctf_pad:-self.ctf_pad, 
                self.ctf_pad:-self.ctf_pad
            ].real
        else:
            return projs_ctf.real
        
if __name__ == "__main__":
    pixel_width = 1.
    sidelength = 64
    ctf_pad = 0
    spherical_aberration = 2.7
    amplitude_contrast_ratio = 0.1
    defocus = 20000
    electron_energy = 300.

    ctf = CTF(pixel_width,sidelength,ctf_pad,spherical_aberration,amplitude_contrast_ratio,defocus,electron_energy)

    