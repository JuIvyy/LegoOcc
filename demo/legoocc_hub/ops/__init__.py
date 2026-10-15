"""Gaussian-to-voxel local aggregation CUDA op.

The extension is compiled just-in-time on first use (via
``torch.utils.cpp_extension.load``) and cached under
``$TORCH_EXTENSIONS_DIR`` (default ``~/.cache/torch_extensions``), so the
build cost (~1-2 min) is only paid once per environment.

Requirements for the JIT build: ``nvcc`` matching the CUDA version of your
PyTorch build (found via ``$CUDA_HOME`` or ``PATH``) and ``ninja``.
"""
import os
import os.path as osp
import shutil
import warnings

import torch

_CSRC = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
_SOURCES = [
    'src/aggregator_impl.cu',
    'src/forward.cu',
    'src/backward.cu',
    'local_aggregate.cu',
    'ext.cpp',
]
_NAME = 'legoocc_local_aggregate'
_EXT = None


def _check_build_env():
    from torch.utils.cpp_extension import CUDA_HOME

    if not torch.cuda.is_available():
        raise RuntimeError('LegoOcc voxelization requires a CUDA device.')
    if CUDA_HOME is None or not osp.exists(osp.join(CUDA_HOME, 'bin', 'nvcc')):
        raise RuntimeError(
            'Could not find nvcc to build the LegoOcc CUDA op. Install the CUDA '
            f'toolkit matching your PyTorch build (torch CUDA {torch.version.cuda}) '
            'and set CUDA_HOME, e.g. `export CUDA_HOME=/usr/local/cuda`.')
    if shutil.which('ninja') is None:
        raise RuntimeError('Building the LegoOcc CUDA op requires ninja: `pip install ninja`.')


def _load_cached():
    """Import a previously built extension without needing nvcc/ninja."""
    import importlib.util
    from torch.utils.cpp_extension import _get_build_directory

    so = osp.join(_get_build_directory(_NAME, verbose=False), _NAME + '.so')
    if not osp.exists(so):
        return None
    if max(osp.getmtime(osp.join(_CSRC, s)) for s in _SOURCES) > osp.getmtime(so):
        return None  # sources changed, rebuild
    spec = importlib.util.spec_from_file_location(_NAME, so)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ext(verbose=False):
    """Return the compiled extension, building it on first call."""
    global _EXT
    if _EXT is not None:
        return _EXT

    try:
        _EXT = _load_cached()
    except Exception:
        _EXT = None
    if _EXT is not None:
        return _EXT

    _check_build_env()
    from torch.utils.cpp_extension import load

    if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
        # Only build for the local GPUs to keep compilation fast.
        archs = sorted({'%d.%d' % torch.cuda.get_device_capability(i)
                        for i in range(torch.cuda.device_count())})
        os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(archs)

    print("[LegoOcc] Compiling CUDA op (first run only, ~1-2 min)...")
    _EXT = load(
        name=_NAME,
        sources=[osp.join(_CSRC, s) for s in _SOURCES],
        extra_cuda_cflags=['-Xcompiler', '-fno-gnu-unique'],
        verbose=verbose,
    )
    return _EXT


def try_load_ext():
    try:
        return load_ext()
    except Exception as e:  # keep the model usable for gaussian prediction
        warnings.warn(f'[LegoOcc] CUDA op unavailable, voxelization disabled: {e}')
        return None


class LocalAggregator(torch.nn.Module):
    """Splat 3D semantic Gaussians onto voxel centers (inference only)."""

    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        # unused at inference, kept so checkpoints load with strict=True
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size

    @torch.no_grad()
    def forward(self, pts, means3D, opas, semantics, scales, cov3D_inv, origin):
        """
        Args:
            pts: (N, 3) voxel centers in world frame.
            means3D, opas, semantics, scales: per-Gaussian params, (G, *).
            cov3D_inv: (G, 3, 3) inverse covariance in world frame.
            origin: (3,) world coordinate of the voxel grid corner.
        Returns:
            logits (N, C), bin_logits (N,), density (N,)
        """
        ext = load_ext()
        points_int = ((pts - origin) / self.grid_size).to(torch.int)
        means3D_int = ((means3D - origin) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0
        assert means3D_int[:, 0].max() < self.H
        assert means3D_int[:, 1].max() < self.W
        assert means3D_int[:, 2].max() < self.D

        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        cov3D_inv = cov3D_inv.flatten(1)[:, [0, 4, 8, 1, 5, 2]]
        if opas.shape[-1] == 1:
            opas = opas.squeeze(-1)

        out = ext.local_aggregate(
            pts.float().contiguous(), points_int.contiguous(),
            means3D.float().contiguous(), means3D_int.contiguous(),
            opas.float().contiguous(), semantics.float().contiguous(),
            radii.contiguous(), cov3D_inv.float().contiguous(),
            self.H, self.W, self.D)
        _, logits, bin_logits, density = out[:4]
        return logits, bin_logits, density
