"""CUDA integration tests using physical optical currents, not normalized images."""
import pytest
import torch
from neurosim_cu_esim import GracaDVSSimulator

pytestmark = pytest.mark.cuda


def frame(value):
    return torch.full((4,4),value,device='cuda',dtype=torch.float64)


def make(**kwargs):
    return GracaDVSSimulator(width=4,height=4,max_events=4096,**kwargs)


def test_static_dark_and_lit_have_no_deterministic_events():
    for current in (0,1e-15,1e-14,1e-12,3e-11):
        s=make()
        s(frame(current),1000)
        assert s(frame(current),2000) is None
        assert torch.isfinite(s.state).all()


def test_first_step_is_not_erased_and_both_polarities_work():
    s=make(refractory_us=0)
    s(frame(1e-14),0)
    on=s(frame(1e-12),10000)
    assert on is not None and (on.p==1).any()
    assert (on.t.to(torch.int64)>0).all() and (on.t.to(torch.int64)<=10000).all()
    off=s(frame(1e-14),50000)
    assert off is not None and (off.p==0).any()
    assert on.x.dtype==torch.uint16 and on.t.dtype==torch.uint64


def test_same_linear_current_trajectory_is_partition_invariant():
    coarse,fine=make(refractory_us=0),make(refractory_us=0)
    for s in (coarse,fine): s(frame(1e-14),0)
    coarse(frame(1e-12),1000)
    for ts in range(10,1001,10):
        fine(frame(1e-14+(1e-12-1e-14)*ts/1000),ts)
    torch.testing.assert_close(coarse.state,fine.state,rtol=1e-8,atol=1e-11)


def test_larger_currents_are_not_clamped_to_one_picoamp():
    s=make()
    s(frame(2e-12),0)
    assert s(frame(4e-12),10000) is not None


def test_seeded_noise_repeats_analog_state():
    states=[]
    for _ in range(2):
        s=make(add_noise=True,seed=12)
        s(frame(1e-14),0)
        s(frame(1e-14),5000)
        states.append(s.state.clone())
    assert torch.equal(*states)
    assert states[0][4].std()>0


def test_nondefault_cuda_stream_and_odd_dimensions():
    s=GracaDVSSimulator(width=33,height=9,max_events=10000)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        s(torch.full((9,33),1e-14,device='cuda'),0)
        events=s(torch.full((9,33),1e-12,device='cuda'),10000)
    stream.synchronize()
    assert events is not None
    assert int(events.x.to(torch.int32).max())<33
    assert int(events.y.to(torch.int32).max())<9


def test_overflow_raises_and_clears_state():
    s=GracaDVSSimulator(width=4,height=4,max_events=1,refractory_us=0)
    s(frame(1e-14),0)
    with pytest.raises(RuntimeError,match='overflow'):
        s(frame(1e-12),10000)
    assert not s.is_initialised
