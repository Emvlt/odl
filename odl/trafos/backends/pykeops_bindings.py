try:
    import pykeops
    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False

from odl.core.space.entry_points import TENSOR_SPACE_IMPLS

__all__ = ('PYKEOPS_AVAILABLE', 'LAZY_TENSOR_IMPLS')

LAZY_TENSOR_IMPLS = {}

if PYKEOPS_AVAILABLE:
    from pykeops.numpy import LazyTensor as NumpyLazyTensor

    LAZY_TENSOR_IMPLS['numpy'] = NumpyLazyTensor

    if 'pytorch' in TENSOR_SPACE_IMPLS:
        from pykeops.torch import LazyTensor as PytorchLazyTensor
        LAZY_TENSOR_IMPLS['pytorch'] = PytorchLazyTensor

    
