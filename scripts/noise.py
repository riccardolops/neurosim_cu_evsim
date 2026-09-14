"""
Noise model per Graca (2024) thesis Eqs 2.29-2.47 and Graca&Delbruck (2025).

Recipe (thesis Sec 2.4, p.67-68):
  Three white Gaussian *current* streams, variance = 4*q*I/(2*ts):
     i_Ipd ~ N(0, 4q*Ipd/(2 ts))   (photodiode + Mfb shot noise, injected at input node)
     i_Ipr ~ N(0, 4q*Ipr/(2 ts))   (Mamp,n + Mamp,p shot noise, injected at output node)
     i_Isf ~ N(0, 4q*Isf/(2 ts))   (SF shot noise, injected at SF output)
  Filter each by its transfer impedance and sum:
     Vpr_noise = Zm(s)*i_Ipd + Zout(s)*i_Ipr
     Vsf_noise = Asf(s)*Vpr_noise + Zoutsf(s)*i_Isf
  No flicker / RTN (thesis: negligible).

Validation target (one-sided) PSD:
     N_Vpr(f)  = 4q*Ipd*|Zm|^2 + 4q*Ipr*|Zout|^2                         (Eq 2.36)
     N_Vsf(f)  = (4q*Ipd*|Zm|^2 + 4q*Ipr*|Zout|^2)*|Asf|^2 + 4q*Isf*|Zoutsf|^2  (Eq 2.44)
"""
import numpy as np
from numpy.random import default_rng
from scipy.signal import bilinear, lfilter, welch, freqs
from model_fast import UT, KAPPA_FB, KAPPA_SF, VA, Q


def analog_tfs(params, Ipd_op, kappa_fb=KAPPA_FB, kappa_sf=KAPPA_SF, VA=VA, UT=UT):
    """Return analog (num,den) for Zm, Zout, Asf, Zoutsf at operating point Ipd_op."""
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    gs_fb = Ipd_op / UT
    gm_fb = kappa_fb * Ipd_op / UT
    gm_amp_n = kappa_fb * Ipr / UT
    Rout = VA / (2.0 * Ipr)
    Aloop = gm_amp_n * Rout * gm_fb / gs_fb
    Rin = 1.0 / gs_fb

    # shared 2nd-order denominator
    a2 = (Rout * (Cpd*Cfb + Cpr*Cfb + Cpr*Cpd)) / (gs_fb * (Aloop + 1.0))
    a1 = (Rin * (Cpd + (1.0 + gm_amp_n*Rout)*Cfb)
          + Rout * (Cpr + (1.0 - gm_fb*Rin)*Cfb)) / (Aloop + 1.0)
    den2 = [a2, a1, 1.0]

    # Zm = Vpr/Ipd
    Zm_dc = (1.0/gm_fb) * (Aloop/(Aloop+1.0))
    wz_Zm = -gm_amp_n / Cfb
    Zm = ([Zm_dc/wz_Zm, Zm_dc], den2)

    # Zout = Vpr/Ipr
    Zout_dc = Rout / (Aloop + 1.0)
    wz_Zout = -gs_fb / (Cpd + Cfb)
    Zout = ([Zout_dc/wz_Zout, Zout_dc], den2)

    # Asf = Vsf/Vpr  and  Zoutsf = Vsf/Isf  (1st order, same pole)
    gs_sf = Isf / UT
    tau_sf = Csf / gs_sf
    Asf = ([kappa_sf], [tau_sf, 1.0])
    Zoutsf = ([1.0/gs_sf], [tau_sf, 1.0])
    return Zm, Zout, Asf, Zoutsf


def gen_noise(params, Ipd_op, ts=10e-6, T=2.0, seed=0):
    """Generate steady-state Vpr,Vsf noise time-series at operating point Ipd_op."""
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    N = int(round(T/ts)); Fs = 1.0/ts
    Zm, Zout, Asf, Zoutsf = analog_tfs(params, Ipd_op)
    Zm_d = bilinear(*Zm, Fs); Zout_d = bilinear(*Zout, Fs)
    Asf_d = bilinear(*Asf, Fs); Zoutsf_d = bilinear(*Zoutsf, Fs)

    rng = default_rng(seed)
    i_ipd = rng.normal(0, np.sqrt(4*Q*Ipd_op/(2*ts)), N)
    i_ipr = rng.normal(0, np.sqrt(4*Q*Ipr/(2*ts)),    N)
    i_isf = rng.normal(0, np.sqrt(4*Q*Isf/(2*ts)),    N)

    vpr_n = lfilter(*Zm_d, i_ipd) + lfilter(*Zout_d, i_ipr)
    vsf_n = lfilter(*Asf_d, vpr_n) + lfilter(*Zoutsf_d, i_isf)
    return dict(vpr=vpr_n, vsf=vsf_n, Fs=Fs)


def theoretical_psd(params, Ipd_op, f):
    """One-sided theoretical PSD (V^2/Hz) at Vpr and Vsf on frequency grid f (Hz)."""
    Zm, Zout, Asf, Zoutsf = analog_tfs(params, Ipd_op)
    w = 2*np.pi*f
    _, Hzm = freqs(*Zm, worN=w);  _, Hzo = freqs(*Zout, worN=w)
    _, Has = freqs(*Asf, worN=w); _, Hzs = freqs(*Zoutsf, worN=w)
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = params
    Npr = 4*Q*Ipd_op*np.abs(Hzm)**2 + 4*Q*Ipr*np.abs(Hzo)**2
    Nsf = Npr*np.abs(Has)**2 + 4*Q*Isf*np.abs(Hzs)**2
    return Npr, Nsf


if __name__ == "__main__":
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    # physically-anchored fitted params (on the Cpd-Cfb valley)
    params = (71.54e-15, 0.87e-15, 23.72e-15, 583e-15, 3e-9, 10e-12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for Ipd_op, col in [(10e-15, 'C0'), (1e-12, 'C1')]:
        n = gen_noise(params, Ipd_op, ts=2e-6, T=4.0, seed=1)
        for ax, key, lab in [(axes[0], 'vpr', 'Vpr'), (axes[1], 'vsf', 'Vsf')]:
            f, Pxx = welch(n[key], fs=n['Fs'], nperseg=2**15, scaling='density')
            Npr, Nsf = theoretical_psd(params, Ipd_op, f[1:])
            Nth = (Npr if key == 'vpr' else Nsf)
            sig = np.sqrt(np.trapezoid(Pxx, f))
            ax.loglog(f, np.sqrt(Pxx), col, lw=1, alpha=.6,
                      label=f'Ipd={Ipd_op*1e15:.0f}fA sim (σ={sig*1e3:.2f}mV)')
            ax.loglog(f[1:], np.sqrt(Nth), col+'', lw=2, ls='--')
            ax.set_title(f'Noise PSD at {lab} (solid=time-domain, dashed=theory)')
            ax.set_xlabel('Hz'); ax.set_ylabel('V/sqrt(Hz)'); ax.grid(alpha=.3, which='both')
            ax.set_xlim(0.1, 1e5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig('noise_psd.png', dpi=110)
    # report sigmas in log-e (TC) units: 1 e-fold ~ kappa_sf*UT/kappa_fb at Vsf
    Vsf_per_efold = KAPPA_SF*UT/KAPPA_FB
    for Ipd_op in (10e-15, 1e-12):
        n = gen_noise(params, Ipd_op, ts=2e-6, T=8.0, seed=3)
        s_pr = np.std(n['vpr']); s_sf = np.std(n['vsf'])
        print(f"Ipd={Ipd_op*1e15:6.0f}fA: σ_Vpr={s_pr*1e3:.3f}mV  σ_Vsf={s_sf*1e3:.3f}mV "
              f"= {s_sf/Vsf_per_efold:.4f} log-e (thesis Fig2.13: 0.043-0.073 log-e)")
    print("saved noise_psd.png")
