"""
CORRECTED + complete DVS pixel model (Graca & Delbruck 2025), Python reference.

Differences vs the original phys_accur_model.m:
  * SF filter state is CONTINUOUS (Vsf = Asf filtering of Vpr) -- the original reset
    of zi_sf at the operating-point change destroyed the slow SF response.
  * Shot-noise model added (thesis Eqs 2.45-2.47): 3 white current streams of
    variance 4qI/(2Ts), filtered by Zm,Zout -> Vpr and Asf,Zoutsf -> Vsf.
  * Event threshold includes the 1/kappa_fb factor (TC in e-folds of Ipd).
Event generation kept as the v2e fixed-threshold model (thesis-consistent).
"""
import numpy as np
from numpy.random import default_rng
from scipy.signal import bilinear, lfilter
from model_fast import _zm_coeffs, make_input, UT, KAPPA_FB, KAPPA_SF, VA, Q
from noise import theoretical_psd


def ou_params_vsf(params, Ipd_op):
    """OU approximation of the Vsf noise at an operating point:
    sigma (stationary std), tau (correlation time = PSD(0)/(4 sigma^2)),
    s2 = 2 sigma^2/tau (diffusion coefficient, for the FPT crossing formula)."""
    f = np.logspace(-2, 6, 6000)
    _, Nsf = theoretical_psd(params, Ipd_op, f)
    sigma2 = np.trapezoid(Nsf, f)
    tau = Nsf[0] / (4.0*sigma2)          # OU match to the low-freq plateau
    return np.sqrt(sigma2), tau, 2.0*sigma2/tau


def _zout_zsf(params, Ipd_op, Fs, kappa_fb=KAPPA_FB, VA=VA, UT=UT):
    """analog->digital Zout (Vpr/Ipr) and Zoutsf (Vsf/Isf) at operating point."""
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    gs_fb = Ipd_op/UT; gm_amp_n = kappa_fb*Ipr/UT; Rout = VA/(2*Ipr)
    Aloop = gm_amp_n*Rout*kappa_fb
    a2 = (Rout*(Cpd*Cfb + Cpr*Cfb + Cpr*Cpd))/(gs_fb*(Aloop+1))
    a1 = ((1/gs_fb)*(Cpd + (1+gm_amp_n*Rout)*Cfb)
          + Rout*(Cpr + (1-kappa_fb)*Cfb))/(Aloop+1)
    Zout_dc = Rout/(Aloop+1); wz = -gs_fb/(Cpd+Cfb)
    Zout_d = bilinear([Zout_dc/wz, Zout_dc], [a2, a1, 1.0], Fs)
    gs_sf = Isf/UT
    Zsf_d = bilinear([1.0/gs_sf], [Csf/gs_sf, 1.0], Fs)
    return Zout_d, gs_sf


def simulate_full(params, Ts=10e-6, T_end=0.08, add_noise=False, seed=0, warmup=4000,
                  TC=0.3, t_refrac=100e-6, Cdiff_ratio=20.0, stochastic_events=False):
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    t, Ipd = make_input(Ts, T_end); N = len(t); Fs = 1/Ts
    trans = list(np.where(np.diff(Ipd) != 0)[0] + 1)
    seg_starts = [1] + trans; seg_ends = trans + [N]

    # ---- SIGNAL ----
    dVpr = np.zeros(N); zi = np.zeros(2)
    for s, e in zip(seg_starts, seg_ends):
        Ipd_op = Ipd[s]
        A = Ipd_op*np.log(Ipd[s]/Ipd[s-1]) if Ipd[s] != Ipd[s-1] else 0.0
        numd, dend = _zm_coeffs(Ipd_op, Ipr, Cpd, Cfb, Cpr, KAPPA_FB, VA, UT, Fs)
        x = np.zeros(e-s);
        if x.size: x[0] = A
        zi = np.zeros(2)                       # PR coeffs change per segment
        y, _ = lfilter(numd, dend, x, zi=zi)
        dVpr[s:e] = y
    Vpr = np.cumsum(dVpr)
    asf = bilinear([KAPPA_SF], [Csf/(Isf/UT), 1.0], Fs)
    Vsf = lfilter(*asf, Vpr)                   # SF state CONTINUOUS (the fix)

    # ---- NOISE (optional) ----
    s2_by_op = {}                                # empirical OU diffusion per op. point
    if add_noise:
        rng = default_rng(seed)
        vpr_n = np.zeros(N)
        for s, e in zip(seg_starts, seg_ends):
            Ipd_op = Ipd[s]; L = e-s
            zm = _zm_coeffs(Ipd_op, Ipr, Cpd, Cfb, Cpr, KAPPA_FB, VA, UT, Fs)
            zout, _ = _zout_zsf(params, Ipd_op, Fs)
            i_ipd = rng.normal(0, np.sqrt(4*Q*Ipd_op/(2*Ts)), L+warmup)
            i_ipr = rng.normal(0, np.sqrt(4*Q*Ipr/(2*Ts)),    L+warmup)
            seg = lfilter(*zm, i_ipd) + lfilter(*zout, i_ipr)
            vpr_n[s:e] = seg[warmup:]
        _, gs_sf = _zout_zsf(params, Ipd[1], Fs)
        zsf = bilinear([1.0/gs_sf], [Csf/gs_sf, 1.0], Fs)
        i_isf = rng.normal(0, np.sqrt(4*Q*Isf/(2*Ts)), N)
        vsf_n = lfilter(*asf, vpr_n) + lfilter(*zsf, i_isf)
        Vpr = Vpr + vpr_n; Vsf = Vsf + vsf_n
        # empirical OU diffusion s2 = Var(d noise)/Ts of Vsf noise, per operating point
        for s, e in zip(seg_starts, seg_ends):
            if e - s > 3:
                s2_by_op[Ipd[s]] = np.var(np.diff(vsf_n[s:e])) / Ts

    # ---- EVENTS ----
    # Direct threshold crossing (v2e) on Vsf, optionally augmented with the stochastic
    # first-passage-time test (Giraudo 1999 / Bibbona 2008) that catches noise crossings
    # occurring BETWEEN samples. Note: ON when Vsf-Vref >= +theta, OFF when <= -theta.
    Vth_sf = TC * KAPPA_SF * UT / KAPPA_FB      # corrected: /kappa_fb (e-folds of Ipd)
    A_diff = -Cdiff_ratio
    rng_ev = default_rng(seed + 777)
    Vdiff = np.zeros(N); events = np.zeros(N)
    Vref = np.mean(Vsf[:int(round(0.01/Ts))]); t_last = -np.inf
    for n in range(1, N):
        if (t[n]-t_last) < t_refrac:
            Vref = Vsf[n]; continue
        d = Vsf[n] - Vref
        Vdiff[n] = A_diff*d
        ev = 0
        if d >= Vth_sf:        ev = 1            # direct ON
        elif d <= -Vth_sf:     ev = -1           # direct OFF
        elif stochastic_events and Ipd[n] in s2_by_op:   # FPT hidden crossing (2 Bernoulli)
            dp = Vsf[n-1] - Vref
            s2dt = s2_by_op[Ipd[n]] * Ts
            if dp < Vth_sf and rng_ev.random() < np.exp(-2*(Vth_sf-dp)*(Vth_sf-d)/s2dt):
                ev = 1
            elif dp > -Vth_sf and rng_ev.random() < np.exp(-2*(dp+Vth_sf)*(d+Vth_sf)/s2dt):
                ev = -1
        if ev != 0:
            events[n] = ev; Vref = Vsf[n]; Vdiff[n] = 0; t_last = t[n]
    return dict(t=t, Ipd=Ipd, Vpr=Vpr, Vsf=Vsf, Vdiff=Vdiff, events=events)


if __name__ == "__main__":
    import scipy.io as sio
    P = (71.54e-15, 0.87e-15, 23.72e-15, 581e-15, 3e-9, 10e-12)
    d = sio.loadmat('paper_data_all.mat')
    r = simulate_full(P, add_noise=True)
    
    # Save reference trace for CUDA validation
    import os
    np.savez("ref_trace.npz", t=r["t"], Ipd=r["Ipd"], Vpr=r["Vpr"], Vsf=r["Vsf"])
    print("saved ref_trace.npz")
    
    epr = np.sqrt(np.mean((np.interp(d['t_vpr'].squeeze(), r['t'], r['Vpr'])-d['vpr'].squeeze())**2))*1e3
    esf = np.sqrt(np.mean((np.interp(d['t_vsf'].squeeze(), r['t'], r['Vsf'])-d['vsf'].squeeze())**2))*1e3
    print(f"signal-only: RMSE Vpr={epr:.2f} mV  Vsf={esf:.2f} mV  "
          f"nON={int((r['events']>0).sum())} nOFF={int((r['events']<0).sum())}")
    rn = simulate_full(P, add_noise=True, seed=2)
    print(f"with noise:  sigma added Vpr={np.std(rn['Vpr']-r['Vpr'])*1e3:.2f} mV  "
          f"Vsf={np.std(rn['Vsf']-r['Vsf'])*1e3:.2f} mV  "
          f"nON={int((rn['events']>0).sum())} nOFF={int((rn['events']<0).sum())}")
