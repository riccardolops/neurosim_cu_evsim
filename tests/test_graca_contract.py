"""CPU interface regressions; the dispatcher is mocked, never claimed as CUDA validation."""
import math
from unittest.mock import Mock
import pytest
import torch
import neurosim_cu_esim.graca as model
from neurosim_cu_esim.graca import GracaDVSSimulator as Simulator


def sim(**kwargs):
    return Simulator(width=2, height=2, device='cpu', **kwargs)


def frame(value=1e-14):
    return torch.full((2, 2), value, dtype=torch.float64)


def empty(*args):
    return tuple(args[i][:0] for i in (4, 5, 6, 7))


def test_first_frame_current_is_retained_and_dark_added():
    s = sim()
    s.init(frame(), timestamp_us=123456)
    assert torch.allclose(s.state[22], frame(1.1e-14), rtol=1e-14, atol=0)
    assert s._prev_time == 123456
    assert s.state.dtype == torch.float64
    assert torch.count_nonzero(s.state[2:8]) == 0
    assert torch.all(s.state[8] == s.refractory_us)


def test_absolute_reference_initializes_physical_equilibrium():
    s = sim(use_first_frame_as_base=False)
    s.init(frame())
    target = s.UT / s.kappa_fb * s.loop_gain_fraction * math.log(11)
    amp_gain = s.kappa_fb*s.VA/(2*s.UT)
    assert torch.allclose(s.state[2], frame(target))
    assert torch.allclose(s.state[3], frame(-target/amp_gain))
    assert torch.allclose(s.state[4], frame(target*s.kappa_sf))
    assert torch.equal(s.state[4], s.state[7])


def test_zero_optical_current_has_dark_operating_point():
    s=sim()
    s.init(frame(0))
    assert torch.equal(s.state[22], frame(s.dark_current))
    assert torch.isfinite(s.state).all()


def test_si_preservation_and_no_default_ceiling():
    s=sim()
    assert s.full_well_saturation_threshold is None
    assert torch.equal(s._prepare_image(frame(4e-12)), frame(4e-12))


def test_explicit_total_current_bound_rejects_instead_of_clipping():
    s=sim(full_well_saturation_threshold=1e-12)
    s.init(frame(0.9e-12))
    with pytest.raises(ValueError, match='ceiling'):
        s.forward(frame(1.1e-12), 10)
    assert s._prev_time == 0
    assert s.is_initialised


def test_extrapolation_diagnostic_once():
    s=sim()
    with pytest.warns(RuntimeWarning, match='Ipr'):
        s.init(frame(1e-9))
    assert s._warned_current


@pytest.mark.parametrize('kwargs', [
    {'Cpd':0}, {'Cfb':-1}, {'Cpr':float('nan')}, {'Csf':-1}, {'Ipr':0},
    {'Isf':float('inf')}, {'dark_current':0}, {'kappa_fb':1.1},
    {'kappa_sf':0}, {'VA':0}, {'UT':-1}, {'contrast_threshold':0},
    {'contrast_threshold_off':-1}, {'refractory_us':-1}, {'dt_us':0},
    {'dt_us':float('nan')}, {'max_events':0}, {'max_events':1.2}, {'seed':-1},
    {'full_well_saturation_threshold':1e-16}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        sim(**kwargs)


@pytest.mark.parametrize('value', [-1e-15, float('nan'), float('inf')])
def test_invalid_current(value):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        sim().init(frame(value))


@pytest.mark.parametrize('value', [-1, 0.5, True, 2**64])
def test_invalid_timestamp(value):
    with pytest.raises(ValueError, match='uint64'):
        sim().forward(frame(), value)


def test_bad_shape_and_integer_current():
    s=sim()
    with pytest.raises(ValueError, match='single-channel'):
        s.init(torch.zeros((3,2)))
    with pytest.raises(ValueError, match='floating-point'):
        s.init(torch.zeros((2,2),dtype=torch.int32))


def test_channel_axis_and_noncontiguous():
    s=sim()
    assert s._prepare_image(frame().unsqueeze(-1)).shape == (2,2)
    assert s._prepare_image(frame().t()).is_contiguous()


def test_first_transition_dispatched_with_initial_current_and_time(monkeypatch):
    dispatcher=Mock(side_effect=empty)
    monkeypatch.setattr(model,'evsim_graca_cuda',dispatcher)
    s=sim(dt_us=5)
    assert s(frame(),1000) is None
    assert s(frame(1e-12),1010) is None
    args=dispatcher.call_args.args
    assert (args[1],args[2]) == (1010,1000)
    assert torch.equal(args[3][22],frame(1.1e-14))
    assert torch.equal(args[0],frame(1e-12))
    assert args[-1]==5
    assert args[-2]==1
    assert s._frame_index == 1


def test_dispatch_error_invalidates_advanced_state(monkeypatch):
    monkeypatch.setattr(model,'evsim_graca_cuda',Mock(side_effect=RuntimeError('overflow')))
    s=sim()
    s(frame(),0)
    with pytest.raises(RuntimeError, match='overflow'):
        s(frame(1e-12),10)
    assert not s.is_initialised


def test_event_batches_do_not_alias_reusable_buffers(monkeypatch):
    def dispatch(*args):
        args[7][0] = 1 if args[1]==10 else 0
        return tuple(args[i][:1] for i in (4,5,6,7))
    monkeypatch.setattr(model,'evsim_graca_cuda',dispatch)
    s=sim()
    s(frame(),0)
    first=s(frame(1e-12),10)
    second=s(frame(),20)
    assert first.p.item()==1 and second.p.item()==0


def test_threshold_mapping_uses_finite_loop_gain():
    s=sim(contrast_threshold=.2,contrast_threshold_off=.4)
    assert s.thr_on == pytest.approx(.004985017087276)
    assert s.thr_off == 2*s.thr_on


def test_reset_clears_and_reinitializes_epoch():
    s=sim()
    s(frame(),100)
    s.reset(frame(2e-14),timestamp_us=500)
    assert s._prev_time==500 and s._frame_index==0
    assert torch.equal(s.state[22],frame(2.1e-14))


def test_unvalidated_fpt_is_explicitly_unavailable():
    with pytest.raises(NotImplementedError,match='FPT'):
        sim(stochastic_events=True)


def test_submicrosecond_analog_step_is_allowed():
    assert sim(dt_us=0.1).dt_us == 0.1


def test_float16_cannot_silently_underflow_photocurrent():
    with pytest.raises(ValueError,match="float16"):
        sim().init(frame().half())
