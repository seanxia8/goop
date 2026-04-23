"""
SIREN-backed TOF sampler.

Drop-in replacement for the voxel-LUT ``TOFSampler``: the PCA basis,
quantile-time reconstruction, Poisson sampling, and differentiable
``sample_pdf`` path are all inherited from ``PCATOFSampler``. Only
``_lookup(pos)`` is reimplemented — instead of trilinear LUT
interpolation, we forward a batch of (position, PMT-position) pairs
through a pre-trained ``PcaSiren`` network that directly predicts
``(vis, log_t0, coeffs)`` in the same PCA basis as the LUT's stored
arrays.

Differentiability: the network runs in ``eval`` mode with frozen
parameters, but its forward is *not* wrapped in ``torch.no_grad`` —
activations stay in the autograd graph so gradients flow from waveforms
back through ``_lookup`` to input positions.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

from .base import DEFAULT_N_SIMULATED, DEFAULT_PLIB_PATH, PCATOFSampler

__all__ = [
    "SirenTOFSampler",
    "create_siren_tof_sampler",
]

DEFAULT_CKPT_PATH = (
    "/sdf/group/neutrino/youngsam/sirentv/logs_pca_81/version-67/"
    "iteration-1558000-epoch-2000.ckpt"
)
DEFAULT_CFG_PATH = (
    "/sdf/group/neutrino/youngsam/sirentv/logs_pca_81/version-67/train_cfg.yaml"
)
DEFAULT_SIRENTV_SRC = "/sdf/home/y/youngsam/sw/dune/sirentv"

_CKPT_PREFIXES = (
    "_orig_mod.model.",
    "_orig_mod.",
    "model.model.",
    "model.",
)


class SirenTOFSampler(PCATOFSampler):
    """TOF sampler whose ``_lookup`` is a pre-trained SIREN network.

    The network (``sirentv.models.pca_siren.PcaSiren``) takes a ``(B, P, 6)``
    tensor of concatenated normalized (source_pos, pmt_pos) pairs and returns
    a dict ``{v, t0, coeffs}`` — the *same* quantities stored in the compressed
    photon library, so every reconstruction/sampling method on ``PCATOFSampler``
    works without modification.
    """

    def __init__(
        self,
        plib_path: str = DEFAULT_PLIB_PATH,
        ckpt_path: str = DEFAULT_CKPT_PATH,
        cfg_path: str = DEFAULT_CFG_PATH,
        sirentv_src: str | None = DEFAULT_SIRENTV_SRC,
        n_simulated: float = DEFAULT_N_SIMULATED,
        device: str | torch.device = "cuda:0",
        pmt_qe: float = 0.12,
        n_photon: float | None = None,
        verbose: bool = False,
    ):
        dev = torch.device(device) if isinstance(device, str) else device

        # 1. shared PCA basis + voxel metadata + PMT positions
        basis = PCATOFSampler._read_h5_basis(plib_path)
        if basis["mode"] != "log_quantile":
            raise ValueError(
                f"SirenTOFSampler expects plib mode='log_quantile', got {basis['mode']!r}"
            )
        if basis["pmt_pos"] is None:
            raise ValueError(f"{plib_path} is missing 'pmt_pos' — required for SIREN input")

        self._init_common(
            device=dev,
            n_simulated=n_simulated,
            pmt_qe=pmt_qe,
            n_pmts=basis["n_pmts"],
            n_components=basis["n_components"],
            log_quantile_C=basis["log_quantile_C"],
            t_max_ns=basis["t_max_ns"],
            mode=basis["mode"],
            pca_mean=basis["pca_mean"],
            pca_components=basis["pca_components"],
            u_grid=basis["u_grid"],
            numvox=basis["numvox"],
            min_xyz=basis["min_xyz"],
            max_xyz=basis["max_xyz"],
        )

        # 2. sirentv imports + network
        if sirentv_src and sirentv_src not in sys.path:
            sys.path.insert(0, sirentv_src)
        from sirentv.models.pca_siren import PcaSiren  # noqa: E402
        from slar.transform import partial_xform_vis  # noqa: E402

        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)

        net_cfg = {
            k: v
            for k, v in cfg["model"]["network"].items()
            if k not in ("type", "use_CDF", "xform_vis")
        }
        net = PcaSiren(**net_cfg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clean_state = {}
        for k, v in state.items():
            kk = k
            for prefix in _CKPT_PREFIXES:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
                    break
            clean_state[kk] = v
        missing, unexpected = net.load_state_dict(clean_state, strict=False)
        if verbose and (missing or unexpected):
            print(
                f"[SirenTOFSampler] load_state_dict: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        net.to(self._device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

        # 3. visibility inverse transform + n_photon
        _, inv_xform_vis = partial_xform_vis(cfg.get("transform_vis", {}))
        self._inv_xform_vis = inv_xform_vis
        if n_photon is None:
            n_photon = cfg.get("compressed_plib", {}).get("n_photon", 1.5e7)
        self._n_photon = float(n_photon)

        # 4. cached normalized PMT positions
        pmt_pos_t = torch.from_numpy(basis["pmt_pos"].astype(np.float32)).to(self._device)
        self._norm_pmt_pos = self._normalize_coord(pmt_pos_t)  # (P, 3) in [-1, 1]

    def _normalize_coord(self, pos: torch.Tensor) -> torch.Tensor:
        """Map world-mm coordinates to [-1, 1] per axis (matches VoxelMeta.norm_coord)."""
        lo = self._min_xyz.to(dtype=pos.dtype, device=pos.device)
        hi = self._max_xyz.to(dtype=pos.dtype, device=pos.device)
        return 2.0 * (pos - lo) / (hi - lo) - 1.0

    def _lookup(self, pos: torch.Tensor):
        """pos: (N, 3) on the x<=0 half-detector -> (vis, t0, coeffs).

        Shapes returned: vis (N, P), t0 (N, P), coeffs (N, P, K) — all float32
        and differentiable w.r.t. ``pos``.
        """
        pos = pos.to(dtype=torch.float32, device=self._device)
        N, P = pos.shape[0], self._n_pmts

        pos_norm = self._normalize_coord(pos)                     # (N, 3)
        src = pos_norm.unsqueeze(1).expand(N, P, 3)               # (N, P, 3)
        pmt = self._norm_pmt_pos.unsqueeze(0).expand(N, P, 3)     # (N, P, 3)
        inp = torch.cat([src, pmt], dim=-1)                       # (N, P, 6)

        out = self.net(inp)  # no detach — activations carry grad to inp -> pos
        vis = self._inv_xform_vis(out["v"]) * self._n_photon       # (N, P)
        t0_ns = torch.exp(out["t0"])                                # (N, P)
        coeffs = out["coeffs"]                                       # (N, P, K)
        return vis, t0_ns, coeffs


def create_siren_tof_sampler(**kwargs) -> SirenTOFSampler:
    """Factory with sensible defaults. See ``SirenTOFSampler.__init__`` for kwargs."""
    defaults = {
        "plib_path": DEFAULT_PLIB_PATH,
        "ckpt_path": DEFAULT_CKPT_PATH,
        "cfg_path": DEFAULT_CFG_PATH,
        "sirentv_src": DEFAULT_SIRENTV_SRC,
        "n_simulated": DEFAULT_N_SIMULATED,
        "device": "cuda:0",
        "pmt_qe": 0.12,
    }
    defaults.update(kwargs)
    return SirenTOFSampler(**defaults)
