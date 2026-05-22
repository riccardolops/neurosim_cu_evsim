"""Tests that exercise the low-level CUDA extension directly."""

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.fixture
def buffers():
    """Create pre-allocated event buffers."""
    max_events = 4096
    dev = "cuda"
    return {
        "x": torch.empty(max_events, dtype=torch.uint16, device=dev),
        "y": torch.empty(max_events, dtype=torch.uint16, device=dev),
        "t": torch.empty(max_events, dtype=torch.uint64, device=dev),
        "p": torch.empty(max_events, dtype=torch.uint8, device=dev),
    }


class TestBackend:
    def test_import(self):
        """The compiled extension should be importable."""
        import _neurosim_cu_esim_ext  # noqa: F401

    def test_direct_call_no_events(self, buffers):
        """Calling with matching state should return empty tensors."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 32, 32
        dev = "cuda"
        img = torch.full((h, w), 0.5, device=dev)
        log_img = torch.log(img)
        ub = (log_img + 0.35).contiguous()
        lb = (log_img - 0.35).contiguous()

        x, _y, _t, _p = evsim_cuda(
            img,
            100,
            ub,
            lb,
            buffers["x"],
            buffers["y"],
            buffers["t"],
            buffers["p"],
            0.35,
            0.35,
        )
        assert x.numel() == 0

    def test_direct_call_with_events(self, buffers):
        """A big brightness jump should produce events."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 32, 32
        dev = "cuda"
        dark = torch.full((h, w), 0.01, device=dev)
        bright = torch.full((h, w), 10.0, device=dev)
        log_dark = torch.log(dark)
        ub = (log_dark + 0.35).contiguous()
        lb = (log_dark - 0.35).contiguous()

        x, _y, t, _p = evsim_cuda(
            bright,
            999,
            ub,
            lb,
            buffers["x"],
            buffers["y"],
            buffers["t"],
            buffers["p"],
            0.35,
            0.35,
        )
        assert x.numel() == h * w
        assert (t == 999).all()

    def test_state_mutation(self, buffers):
        """The state tensors should be mutated in-place by the kernel."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 16, 16
        dev = "cuda"
        img = torch.full((h, w), 0.5, device=dev)
        log_img = torch.log(img)
        ub = (log_img + 0.35).contiguous()
        lb = (log_img - 0.35).contiguous()

        ub_before = ub.clone()
        lb_before = lb.clone()

        bright = torch.full((h, w), 10.0, device=dev)
        evsim_cuda(
            bright,
            1,
            ub,
            lb,
            buffers["x"],
            buffers["y"],
            buffers["t"],
            buffers["p"],
            0.35,
            0.35,
        )

        # States should have changed
        assert not torch.equal(ub, ub_before)
        assert not torch.equal(lb, lb_before)

    def test_non_contiguous_raises(self, buffers):
        """Non-contiguous input should be rejected."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 32, 32
        dev = "cuda"
        img = torch.full((h, w), 0.5, device=dev)
        log_img = torch.log(img)
        ub = (log_img + 0.35).contiguous()
        lb = (log_img - 0.35).contiguous()

        non_contig = img.t()  # transpose makes it non-contiguous
        with pytest.raises(RuntimeError, match="contiguous"):
            evsim_cuda(
                non_contig,
                1,
                ub,
                lb,
                buffers["x"],
                buffers["y"],
                buffers["t"],
                buffers["p"],
                0.35,
                0.35,
            )

    def test_cpu_tensor_raises(self, buffers):
        """CPU tensors should be rejected."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 8, 8
        img_cpu = torch.full((h, w), 0.5)
        ub = torch.full((h, w), 0.0)
        lb = torch.full((h, w), 0.0)

        with pytest.raises(RuntimeError, match="CUDA"):
            evsim_cuda(
                img_cpu,
                1,
                ub,
                lb,
                buffers["x"],
                buffers["y"],
                buffers["t"],
                buffers["p"],
                0.35,
                0.35,
            )

    def test_wrong_dim_raises(self, buffers):
        """3-D image tensor should be rejected by the backend."""
        from neurosim_cu_esim._backend import evsim_cuda

        dev = "cuda"
        img_3d = torch.full((1, 8, 8), 0.5, device=dev)
        ub = torch.full((8, 8), 0.0, device=dev)
        lb = torch.full((8, 8), 0.0, device=dev)

        with pytest.raises(RuntimeError, match="2-D"):
            evsim_cuda(
                img_3d,
                1,
                ub,
                lb,
                buffers["x"],
                buffers["y"],
                buffers["t"],
                buffers["p"],
                0.35,
                0.35,
            )

    def test_multi_emits_multiple_events_per_pixel(self, buffers):
        """evsim_multi should emit several events for a large log jump."""
        import math

        from neurosim_cu_esim._backend import evsim_multi_cuda

        h, w = 8, 8
        dev = "cuda"
        ct = 0.35
        v0 = 1.0
        v1 = math.exp(3.5 * ct)  # -> 3 events per pixel
        img = torch.full((h, w), v1, device=dev)
        log0 = math.log(v0)
        ub = torch.full((h, w), log0 + ct, device=dev)
        lb = torch.full((h, w), log0 - ct, device=dev)

        x, _y, t, p = evsim_multi_cuda(
            img,
            2000,  # new_time
            1000,  # prev_time
            ub,
            lb,
            buffers["x"],
            buffers["y"],
            buffers["t"],
            buffers["p"],
            ct,
            ct,
        )
        assert x.numel() == h * w * 3
        assert (p == 1).all()
        # timestamps span (1000, 2000], last == new_time
        assert t.to(torch.int64).max().item() == 2000
        assert t.to(torch.int64).min().item() > 1000

    def test_multi_rejects_prev_after_new(self, buffers):
        """prev_time > new_time should be rejected."""
        from neurosim_cu_esim._backend import evsim_multi_cuda

        h, w = 8, 8
        dev = "cuda"
        img = torch.full((h, w), 1.0, device=dev)
        ub = torch.full((h, w), 0.35, device=dev)
        lb = torch.full((h, w), -0.35, device=dev)
        with pytest.raises(RuntimeError, match="prev_time"):
            evsim_multi_cuda(
                img,
                1000,
                2000,
                ub,
                lb,
                buffers["x"],
                buffers["y"],
                buffers["t"],
                buffers["p"],
                0.35,
                0.35,
            )

    def test_double_precision(self, buffers):
        """The kernel should work with float64 images."""
        from neurosim_cu_esim._backend import evsim_cuda

        h, w = 16, 16
        dev = "cuda"
        dark = torch.full((h, w), 0.01, device=dev, dtype=torch.float64)
        bright = torch.full((h, w), 10.0, device=dev, dtype=torch.float64)
        log_dark = torch.log(dark)
        ub = (log_dark + 0.35).contiguous()
        lb = (log_dark - 0.35).contiguous()

        x, _y, _t, _p = evsim_cuda(
            bright,
            1,
            ub,
            lb,
            buffers["x"],
            buffers["y"],
            buffers["t"],
            buffers["p"],
            0.35,
            0.35,
        )
        assert x.numel() == h * w
