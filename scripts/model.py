"""
Faithful Python port of phys_accur_model.m (Graca & Delbruck 2025 DVS pixel model).

scipy.signal.bilinear  == MATLAB bilinear(num,den,Fs)   (Tustin, s -> (2/T)(z-1)/(z+1))
scipy.signal.lfilter   == MATLAB filter(b,a,x,zi)        (Direct-Form II Transposed, same state)

This module is used to (1) verify the implementation against the paper data and
(2) fit the 4 capacitances + 2 bias currents.  Compute happens here because no
MATLAB/octave runtime is available; the fitted result is ported back to the .m file.
"""
import numpy as np
from scipy.signal import bilinear, lfilter

# ---- fixed physical constants (same as the .m file) ----
Q   = 1.602e-19
UT  = 25.8e-3       # thermal voltage ~300 K
KAPPA_FB = 0.7      # subthreshold slope (feedback + amp)  -- paper uses kappa_amp,n separately
KAPPA_SF = 0.7      # subthreshold slope (source follower)
VA  = 3.0           # Early voltage (both amp transistors assumed equal)


def make_input(Ts=10e-6, T_end=0.08, pulse_start=0.01, pulse_end=0.011,
               I_lo=10e-15, I_hi=1e-12):
    """Reproduce the 1 ms 10 fA -> 1 pA -> 10 fA pulse on the 0:Ts:T_end grid."""
    t = np.arange(0.0, T_end + Ts/2, Ts)          # mimic MATLAB 0:Ts:T_end (inclusive)
    Ipd = np.full(t.shape, I_lo)
    Ipd[(t >= pulse_start) & (t <= pulse_end)] = I_hi
    return t, Ipd


def simulate(params, Ts=10e-6, T_end=0.08, kappa_fb=KAPPA_FB, kappa_sf=KAPPA_SF,
             VA=VA, UT=UT, pulse=(0.01, 0.011), Ipulse=(10e-15, 1e-12)):
    """
    params = (Cpd, Cfb, Cpr, Csf, Ipr, Isf)
    Returns dict with t, Ipd, Vpr, Vsf (large-signal incremental accumulation,
    exactly as in phys_accur_model.m).
    """
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    t, Ipd = make_input(Ts, T_end, pulse[0], pulse[1], Ipulse[0], Ipulse[1])
    N = len(t)
    Fs = 1.0 / Ts

    delta_Ipd = np.zeros(N)
    delta_Vpr = np.zeros(N)
    delta_Vsf = np.zeros(N)
    Vpr = np.zeros(N)
    Vsf = np.zeros(N)

    zi_pr = np.zeros(2)
    zi_sf = np.zeros(1)

    for n in range(1, N):
        Ipd_op = Ipd[n]

        if Ipd[n] == Ipd[n-1]:
            delta_Ipd[n] = 0.0
        else:
            delta_Ipd[n] = Ipd_op * np.log(Ipd[n] / Ipd[n-1])
            zi_pr = np.zeros(2)        # reset filter state on operating-point change
            zi_sf = np.zeros(1)

        # --- small-signal PR parameters at current operating point ---
        gs_fb = Ipd_op / UT
        gm_fb = kappa_fb * Ipd_op / UT
        gm_amp_n = kappa_fb * Ipr / UT
        Rout = VA / (2.0 * Ipr)
        Aloop = gm_amp_n * Rout * gm_fb / gs_fb
        Rin = 1.0 / gs_fb

        K = (1.0 / gm_fb) * (Aloop / (Aloop + 1.0))
        wz = -gm_amp_n / Cfb
        b0 = K
        b1 = K / wz

        a2 = (Rout * (Cpd*Cfb + Cpr*Cfb + Cpr*Cpd)) / (gs_fb * (Aloop + 1.0))
        a1 = (Rin * (Cpd + (1.0 + gm_amp_n*Rout)*Cfb)
              + Rout * (Cpr + (1.0 - gm_fb*Rin)*Cfb)) / (Aloop + 1.0)

        numd, dend = bilinear([b1, b0], [a2, a1, 1.0], Fs)
        y, zi_pr = lfilter(numd, dend, [delta_Ipd[n]], zi=zi_pr)
        delta_Vpr[n] = y[0]
        Vpr[n] = Vpr[n-1] + delta_Vpr[n]

        # --- source follower ---
        gs_sf = Isf / UT
        a_sf1 = Csf / gs_sf
        numd_sf, dend_sf = bilinear([kappa_sf], [a_sf1, 1.0], Fs)
        y2, zi_sf = lfilter(numd_sf, dend_sf, [delta_Vpr[n]], zi=zi_sf)
        delta_Vsf[n] = y2[0]
        Vsf[n] = Vsf[n-1] + delta_Vsf[n]

    return dict(t=t, Ipd=Ipd, Vpr=Vpr, Vsf=Vsf,
                delta_Ipd=delta_Ipd, delta_Vpr=delta_Vpr, delta_Vsf=delta_Vsf)


if __name__ == "__main__":
    import scipy.io as sio
    # user's current "guessed" parameters
    p0 = (15.93e-15, 1.26e-15, 4.88e-15, 749.83e-15, 95.91e-9, 0.11e-9)
    r = simulate(p0)
    d = sio.loadmat('paper_data_all.mat')
    print("sim Vpr peak = %.4f V (ref %.4f)" % (r['Vpr'].max(), d['vpr'].max()))
    print("sim Vsf peak = %.4f V (ref %.4f)" % (r['Vsf'].max(), d['vsf'].max()))
    print("N steps      = %d" % len(r['t']))
