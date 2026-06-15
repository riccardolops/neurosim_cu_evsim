"""
Vectorized equivalent of phys_accur_model.m, exploiting that Ipd is piecewise
constant -> the small-signal coefficients are constant within each segment, and
the effective log-current input is a single impulse at each transition.

This reproduces the per-sample loop EXACTLY (validated in __main__) but ~100x
faster, which is needed for global parameter optimization.
"""
import numpy as np
from scipy.signal import bilinear, lfilter

Q, UT, KAPPA_FB, KAPPA_SF, VA = 1.602e-19, 25.8e-3, 0.7, 0.7, 3.0


def make_input(Ts=10e-6, T_end=0.08, pulse=(0.01, 0.011), Ipulse=(10e-15, 1e-12)):
    t = np.arange(0.0, T_end + Ts/2, Ts)
    Ipd = np.full(t.shape, Ipulse[0])
    Ipd[(t >= pulse[0]) & (t <= pulse[1])] = Ipulse[1]
    return t, Ipd


def _zm_coeffs(Ipd_op, Ipr, Cpd, Cfb, Cpr, kappa_fb, VA, UT, Fs):
    gs_fb = Ipd_op / UT
    gm_fb = kappa_fb * Ipd_op / UT
    gm_amp_n = kappa_fb * Ipr / UT
    Rout = VA / (2.0 * Ipr)
    Aloop = gm_amp_n * Rout * gm_fb / gs_fb
    Rin = 1.0 / gs_fb
    K = (1.0 / gm_fb) * (Aloop / (Aloop + 1.0))
    wz = -gm_amp_n / Cfb
    b0, b1 = K, K / wz
    a2 = (Rout * (Cpd*Cfb + Cpr*Cfb + Cpr*Cpd)) / (gs_fb * (Aloop + 1.0))
    a1 = (Rin * (Cpd + (1.0 + gm_amp_n*Rout)*Cfb)
          + Rout * (Cpr + (1.0 - gm_fb*Rin)*Cfb)) / (Aloop + 1.0)
    return bilinear([b1, b0], [a2, a1, 1.0], Fs)


def simulate(params, Ts=10e-6, T_end=0.08, kappa_fb=KAPPA_FB, kappa_sf=KAPPA_SF,
             VA=VA, UT=UT, pulse=(0.01, 0.011), Ipulse=(10e-15, 1e-12)):
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    t, Ipd = make_input(Ts, T_end, pulse, Ipulse)
    N = len(t)
    Fs = 1.0 / Ts

    # transition indices (where Ipd changes)
    trans = np.where(np.diff(Ipd) != 0)[0] + 1
    bounds = [1] + list(trans) + [N]   # segment starts (n=0 has no delta)

    # SF coefficients are constant (depend only on Isf, Csf)
    gs_sf = Isf / UT
    numd_sf, dend_sf = bilinear([kappa_sf], [Csf/gs_sf, 1.0], Fs)

    delta_Vpr = np.zeros(N)
    delta_Vsf = np.zeros(N)

    seg_starts = [1] + list(trans)
    seg_ends = list(trans) + [N]
    for s, e in zip(seg_starts, seg_ends):
        Ipd_op = Ipd[s]
        A = Ipd_op * np.log(Ipd[s] / Ipd[s-1]) if Ipd[s] != Ipd[s-1] else 0.0
        numd, dend = _zm_coeffs(Ipd_op, Ipr, Cpd, Cfb, Cpr, kappa_fb, VA, UT, Fs)
        x = np.zeros(e - s);
        if x.size: x[0] = A
        delta_Vpr[s:e] = lfilter(numd, dend, x)          # state resets each segment
        delta_Vsf[s:e] = lfilter(numd_sf, dend_sf, delta_Vpr[s:e])

    Vpr = np.cumsum(delta_Vpr)
    Vsf = np.cumsum(delta_Vsf)
    return dict(t=t, Ipd=Ipd, Vpr=Vpr, Vsf=Vsf)


if __name__ == "__main__":
    import time, model as slow
    p0 = (15.93e-15, 1.26e-15, 4.88e-15, 749.83e-15, 95.91e-9, 0.11e-9)

    t0 = time.time(); rs = slow.simulate(p0); ts = time.time()-t0
    t0 = time.time(); rf = simulate(p0);      tf = time.time()-t0

    dpr = np.max(np.abs(rs['Vpr'] - rf['Vpr']))
    dsf = np.max(np.abs(rs['Vsf'] - rf['Vsf']))
    print(f"slow loop: {ts*1000:7.1f} ms   fast: {tf*1000:6.2f} ms   speedup {ts/tf:.0f}x")
    print(f"max|dVpr| = {dpr:.3e} V   max|dVsf| = {dsf:.3e} V   (should be ~1e-16)")
