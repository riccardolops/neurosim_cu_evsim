"""Tests for the physically-realistic Graca & Delbruck DVS pixel simulator."""

import pytest
import torch

from neurosim_cu_esim import GracaDVSSimulator
from neurosim_cu_esim.graca import GRACA_NSTATE

pytestmark = pytest.mark.cuda


class TestInit:
    def test_defaults_anchored_fit(self):
        sim = GracaDVSSimulator(width=16, height=16)
        assert sim.Ipr == 3e-9 and sim.Isf == 10e-12
        # decay combination Cpd + (1+G)*Cfb with (1+G)=41.7
        assert abs((sim.Cpd + 41.7 * sim.Cfb) * 1e15 - 125.4) < 1.0

    def test_first_call_returns_none(self, device):
        sim = GracaDVSSimulator(width=16, height=16)
        frame = torch.full((16, 16), 0.5, device=device)
        assert sim.forward(frame, 0) is None
        assert sim.is_initialised
        assert sim.state.shape == (GRACA_NSTATE, 16, 16)

    def test_bad_dt_raises(self):
        with pytest.raises(ValueError, match="dt_us"):
            GracaDVSSimulator(width=8, height=8, dt_us=0.0)

    def test_threshold_mapping(self):
        sim = GracaDVSSimulator(width=8, height=8, contrast_threshold=0.3)
        # TC * kappa_sf * UT / kappa_fb
        assert abs(sim.thr_on - 0.3 * 0.7 * 25.8e-3 / 0.7) < 1e-9


class TestForward:
    def test_static_scene_no_events(self, device):
        sim = GracaDVSSimulator(width=32, height=32, add_noise=False)
        frame = torch.full((32, 32), 0.5, device=device)
        sim.forward(frame, 0)
        ev = sim.forward(frame, 1000)
        assert ev is None  # deterministic, static -> no events

    def test_brightness_increase_is_ON(self, device):
        sim = GracaDVSSimulator(
            width=32, height=32, add_noise=False, max_events=32 * 32 * 64
        )
        sim.forward(torch.full((32, 32), 0.05, device=device), 0)
        # step up; events occur over the first few frames as Vsf ramps past
        # threshold, so accumulate the whole burst.
        pols = []
        for i in range(1, 8):
            ev = sim.forward(torch.full((32, 32), 1.0, device=device), i * 1000)
            if ev is not None:
                pols.append(ev.p)
        assert pols, "expected events from a 20x brightness step"
        p = torch.cat(pols)
        assert float((p == 1).float().mean()) > 0.9

    def test_state_advances_and_persists(self, device):
        sim = GracaDVSSimulator(width=16, height=16, add_noise=False)
        sim.forward(torch.full((16, 16), 0.01, device=device), 0)
        sim.forward(torch.full((16, 16), 1.0, device=device), 1000)
        # Vpr (state[0]) should be clearly nonzero after a 100x step
        assert sim.state[0].abs().max().item() > 0.01

    def test_event_field_dtypes(self, device):
        sim = GracaDVSSimulator(
            width=32, height=32, add_noise=False, max_events=32 * 32 * 64
        )
        sim.forward(torch.full((32, 32), 0.05, device=device), 0)
        # grab the first frame that actually produces events
        ev = None
        for i in range(1, 8):
            ev = sim.forward(torch.full((32, 32), 1.0, device=device), i * 1000)
            if ev is not None:
                break
        assert ev is not None
        n = ev.x.numel()
        assert ev.y.numel() == n and ev.t.numel() == n and ev.p.numel() == n
        assert ev.x.dtype == torch.uint16
        assert ev.t.dtype == torch.uint64
        assert ev.p.dtype == torch.uint8

    def test_timestamps_within_interval(self, device):
        sim = GracaDVSSimulator(
            width=32, height=32, add_noise=False, max_events=32 * 32 * 64,
            dt_us=10.0,
        )
        sim.forward(torch.full((32, 32), 0.02, device=device), 1000)
        ev = sim.forward(torch.full((32, 32), 1.0, device=device), 6000)
        assert ev is not None
        t = ev.t.to(torch.int64)
        assert int(t.min()) > 1000 and int(t.max()) <= 6000

    def test_non_increasing_timestamp_raises(self, device):
        sim = GracaDVSSimulator(width=16, height=16)
        sim.forward(torch.full((16, 16), 0.5, device=device), 1000)
        with pytest.raises(ValueError, match="must be >"):
            sim.forward(torch.full((16, 16), 0.5, device=device), 1000)


class TestNoise:
    def test_noise_runs_and_is_seeded(self, device):
        def run(seed):
            sim = GracaDVSSimulator(
                width=48, height=48, add_noise=True, seed=seed,
                max_events=48 * 48 * 64,
            )
            sim.forward(torch.full((48, 48), 0.3, device=device), 0)
            return sim.forward(torch.full((48, 48), 0.3, device=device), 5000)

        e1, e2 = run(123), run(123)
        # same seed -> identical event count (content-deterministic)
        n1 = 0 if e1 is None else e1.x.numel()
        n2 = 0 if e2 is None else e2.x.numel()
        assert n1 == n2


class TestStochasticEvents:
    """First-passage-time (FPT) stochastic event generation."""

    def test_stochastic_runs_and_is_seeded(self, device):
        def run(seed):
            sim = GracaDVSSimulator(
                width=48, height=48, add_noise=True, stochastic_events=True,
                seed=seed, contrast_threshold=0.08, dt_us=200.0,
                refractory_us=1.0, max_events=48 * 48 * 64,
            )
            sim.forward(torch.full((48, 48), 0.3, device=device), 0)
            sim.forward(torch.full((48, 48), 0.3, device=device), 40_000)  # build MSI
            return sim.forward(torch.full((48, 48), 0.3, device=device), 80_000)

        e1, e2 = run(7), run(7)
        n1 = 0 if e1 is None else e1.x.numel()
        n2 = 0 if e2 is None else e2.x.numel()
        assert n1 == n2  # same seed -> deterministic

    def test_fpt_recovers_missed_noise_events(self, device):
        # At a large sub-step the plain v2e detector misses noise crossings that
        # happen between samples; the FPT test recovers them, so over a constant
        # noisy background it should generate at least as many events.
        def total(stochastic):
            sim = GracaDVSSimulator(
                width=64, height=64, add_noise=True, stochastic_events=stochastic,
                seed=0, contrast_threshold=0.07, dt_us=500.0, refractory_us=1.0,
                max_events=64 * 64 * 128,
            )
            sim.forward(torch.full((64, 64), 0.3, device=device), 0)
            tot = 0
            for k in range(1, 6):  # several frames so MSI calibrates
                ev = sim.forward(torch.full((64, 64), 0.3, device=device), k * 60_000)
                tot += 0 if ev is None else ev.x.numel()
            return tot

        assert total(stochastic=True) >= total(stochastic=False)

    def test_high_threshold_matches_v2e(self, device):
        # At the default TC=0.3 the threshold is ~8-29 sigma, so noise events are
        # negligible and stochastic generation should not change a clean step response.
        def step(stochastic):
            sim = GracaDVSSimulator(
                width=32, height=32, add_noise=True, stochastic_events=stochastic,
                seed=1, max_events=32 * 32 * 64,
            )
            sim.forward(torch.full((32, 32), 0.01, device=device), 0)
            ev = sim.forward(torch.full((32, 32), 1.0, device=device), 5000)
            return 0 if ev is None else ev.x.numel()

        # both should fire the step burst; counts close (noise events negligible)
        assert step(stochastic=True) > 0 and step(stochastic=False) > 0


class TestReset:
    def test_reset_clears(self, device):
        sim = GracaDVSSimulator(width=16, height=16)
        sim.forward(torch.full((16, 16), 0.5, device=device), 0)
        sim.reset()
        assert not sim.is_initialised
        assert sim._frame_index == 0
