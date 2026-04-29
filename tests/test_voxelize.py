"""Unit tests for ``goop.utils.voxelize`` — directly exercises the
photon-yield-preserving voxelization step that ``production/run_batch.py``
and the differentiable simulator both depend on.

These cover the invariants the ``notebooks/voxelization_study.ipynb`` and
``notebooks/oversampling_study.ipynb`` notebooks rely on (notably exact
photon-yield preservation across dx values), plus the numpy↔torch dispatch
that ``goop/utils.py:voxelize`` implements.
"""

import numpy as np
import pytest
import torch

from goop.utils import voxelize


def _mk_segments(n=200, seed=0):
    """Random segment cloud roughly the shape of a real edepsim event."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1500.0, 1500.0, size=(n, 3)).astype(np.float32)
    nph = rng.integers(1, 5000, size=(n,), dtype=np.int64)
    tns = rng.uniform(0.0, 10_000.0, size=(n,)).astype(np.float32)
    return pos, nph, tns


class TestVoxelizeBasics:
    def test_photon_yield_preserved(self):
        """Total photon yield must be exactly preserved at any dx > 0.
        This is the load-bearing invariant for production benchmarks."""
        pos, nph, tns = _mk_segments(n=500, seed=1)
        for dx in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            _, nph_v, _ = voxelize(pos, nph, tns, dx=dx)
            assert int(nph_v.sum()) == int(nph.sum()), \
                f"dx={dx}: yield {nph_v.sum()} != {nph.sum()}"

    def test_voxel_count_decreases_with_dx(self):
        """Increasing dx should monotonically reduce voxel count."""
        pos, nph, tns = _mk_segments(n=500, seed=2)
        prev = float("inf")
        for dx in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            pos_v, _, _ = voxelize(pos, nph, tns, dx=dx)
            assert pos_v.shape[0] <= prev
            prev = pos_v.shape[0]

    def test_n_voxels_le_n_segments(self):
        pos, nph, tns = _mk_segments(n=300, seed=3)
        pos_v, nph_v, tns_v = voxelize(pos, nph, tns, dx=10.0)
        assert pos_v.shape[0] <= pos.shape[0]
        assert nph_v.shape[0] == pos_v.shape[0]
        assert tns_v.shape[0] == pos_v.shape[0]

    def test_invalid_dx_raises(self):
        pos, nph, tns = _mk_segments(n=10)
        with pytest.raises(ValueError):
            voxelize(pos, nph, tns, dx=0.0)
        with pytest.raises(ValueError):
            voxelize(pos, nph, tns, dx=-1.0)


class TestVoxelizeDispatch:
    """numpy in → numpy out; torch in → torch out on the input's device."""

    def test_numpy_in_numpy_out(self):
        pos, nph, tns = _mk_segments(n=100)
        pos_v, nph_v, tns_v = voxelize(pos, nph, tns, dx=10.0)
        assert isinstance(pos_v, np.ndarray)
        assert isinstance(nph_v, np.ndarray)
        assert isinstance(tns_v, np.ndarray)
        assert pos_v.dtype == np.float32
        assert tns_v.dtype == np.float32

    def test_torch_cpu_in_torch_cpu_out(self):
        pos, nph, tns = _mk_segments(n=100)
        pos_t = torch.from_numpy(pos)
        nph_t = torch.from_numpy(nph)
        tns_t = torch.from_numpy(tns)
        pos_v, nph_v, tns_v = voxelize(pos_t, nph_t, tns_t, dx=10.0)
        assert isinstance(pos_v, torch.Tensor)
        assert isinstance(nph_v, torch.Tensor)
        assert isinstance(tns_v, torch.Tensor)
        assert pos_v.device.type == "cpu"
        assert pos_v.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
    def test_torch_gpu_in_torch_gpu_out(self):
        pos, nph, tns = _mk_segments(n=100)
        pos_t = torch.from_numpy(pos).cuda()
        nph_t = torch.from_numpy(nph).cuda()
        tns_t = torch.from_numpy(tns).cuda()
        pos_v, nph_v, tns_v = voxelize(pos_t, nph_t, tns_t, dx=10.0)
        assert pos_v.is_cuda
        assert nph_v.is_cuda
        assert tns_v.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
    def test_cpu_and_gpu_paths_agree(self):
        """numpy (CPU numpy path) and torch GPU (torch path) must produce
        bit-equal results — this is the equivalence the production benchmark
        relies on when comparing pre/post GPU-voxelize speedups."""
        pos, nph, tns = _mk_segments(n=300, seed=4)
        # numpy path
        p_n, n_n, t_n = voxelize(pos, nph, tns, dx=10.0)
        # torch GPU path
        p_t, n_t, t_t = voxelize(
            torch.from_numpy(pos).cuda(),
            torch.from_numpy(nph).cuda(),
            torch.from_numpy(tns).cuda(),
            dx=10.0,
        )
        # Sort both by position for comparison (voxel ordering may differ).
        def _sort(p, n, t):
            p = np.asarray(p); n = np.asarray(n); t = np.asarray(t)
            idx = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
            return p[idx], n[idx], t[idx]
        p_n_s, n_n_s, t_n_s = _sort(p_n, n_n, t_n)
        p_t_s, n_t_s, t_t_s = _sort(p_t.cpu().numpy(), n_t.cpu().numpy(), t_t.cpu().numpy())
        assert p_n_s.shape == p_t_s.shape
        assert np.allclose(p_n_s, p_t_s, atol=1e-3)
        assert np.array_equal(n_n_s, n_t_s)
        assert np.allclose(t_n_s, t_t_s, atol=1e-3)


class TestVoxelizeEdgeCases:
    def test_single_segment(self):
        pos = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        nph = np.array([1234], dtype=np.int64)
        tns = np.array([42.0], dtype=np.float32)
        pos_v, nph_v, tns_v = voxelize(pos, nph, tns, dx=10.0)
        assert pos_v.shape == (1, 3)
        assert int(nph_v[0]) == 1234

    def test_all_segments_in_one_voxel(self):
        """All segments at near-identical positions → single voxel,
        with summed photons and weighted-mean position/time."""
        pos = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
                       dtype=np.float32)
        nph = np.array([100, 200, 700], dtype=np.int64)
        tns = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        pos_v, nph_v, tns_v = voxelize(pos, nph, tns, dx=20.0)
        assert pos_v.shape[0] == 1
        assert int(nph_v[0]) == 1000
        # Weighted mean time: (100*10 + 200*20 + 700*30) / 1000 = 26.0
        assert abs(float(tns_v[0]) - 26.0) < 1e-3

    def test_segments_far_apart_do_not_merge(self):
        pos = np.array([[0.0, 0.0, 0.0], [500.0, 500.0, 500.0]],
                       dtype=np.float32)
        nph = np.array([100, 200], dtype=np.int64)
        tns = np.array([10.0, 20.0], dtype=np.float32)
        pos_v, nph_v, tns_v = voxelize(pos, nph, tns, dx=10.0)
        assert pos_v.shape[0] == 2
        assert int(nph_v.sum()) == 300
