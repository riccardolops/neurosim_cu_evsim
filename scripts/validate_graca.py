"""Deterministic CUDA parity against an independent NumPy nodal circuit.

Run after building the extension: python scripts/validate_graca.py
Requires numpy and CUDA PyTorch. Uses physical currents; no fitted trace file or
noisy realization is compared against a deterministic trajectory.
"""
import numpy as np
import torch
from neurosim_cu_esim import GracaDVSSimulator


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: no GPU parity result was produced.')
    sim = GracaDVSSimulator(width=2,height=2,dt_us=10,add_noise=False,max_events=1024)
    times = np.arange(0, 80001, 10, dtype=np.int64)
    photo = np.full(times.shape,9e-15)  # +1 fA dark = paper 10 fA total baseline
    photo[(times>=10000)&(times<11000)] = 999e-15  # 1 pA total
    cpd,cfb,cpr,csf=sim.Cpd,sim.Cfb,sim.Cpr,sim.Csf
    mass=np.array([[cpd+cfb,-cfb,0],[-cfb,cpr+cfb,0],[0,0,csf]])
    state=np.zeros(3)
    gain=sim.UT/sim.kappa_fb*sim.loop_gain_fraction
    previous_target=0.
    reference=np.zeros((len(times),3))
    observed=np.zeros((len(times),3))
    counts=[0,0]
    for index,(time,current) in enumerate(zip(times,photo)):
        events=sim(torch.full((2,2),float(current),dtype=torch.float64,device='cuda'),int(time))
        if index:
            total=current+sim.dark_current
            gs=total/sim.UT;gm=sim.kappa_fb*gs
            ga=sim.kappa_fb*sim.Ipr/sim.UT;rout=sim.VA/(2*sim.Ipr)
            gsf=sim.Isf/sim.UT
            conductance=np.array([[gs,-gm,0],[ga,1/rout,0],[0,-sim.kappa_sf*gsf,gsf]])
            target=gain*np.log(total/(photo[0]+sim.dark_current))
            zmdc=sim.loop_gain_fraction/gm
            forcing=np.array([-(previous_target+target)/(2*zmdc),0,0])
            dt=1e-5
            state=np.linalg.solve(mass+dt/2*conductance,(mass-dt/2*conductance)@state+dt*forcing)
            previous_target=target
            reference[index]=state
        observed[index]=[float(sim.state[k,0,0]) for k in (3,2,4)]
        if events is not None:
            counts[0]+=int((events.p==0).sum())
            counts[1]+=int((events.p==1).sum())
    error=np.max(np.abs(observed-reference),axis=0)
    print('max absolute voltage errors [Vpd,Vpr,Vsf]:',error)
    print('total events [OFF,ON]:',counts)
    assert np.max(error)<2e-9, error
    assert all(x>0 for x in counts)
    print('CUDA deterministic parity PASS (not a silicon/noise calibration).')


if __name__ == '__main__':
    main()
