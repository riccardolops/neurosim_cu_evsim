"""Overlay the CUDA array-model trace (cuda_trace.npz, fetched from artemide)
against the Python single-pixel reference (ref_trace.npz)."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ref = np.load('ref_trace.npz')
cu = np.load('cuda_trace.npz')
t = ref['t']

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, ref['Vpr'], 'k', lw=1.6, label='Python ref (float64)')
ax[0].plot(cu['t'], cu['Vpr'], 'r--', lw=1.1, label='CUDA array (float32)')
ax[0].set_ylabel('Vpr (V)'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title('CUDA array model vs single-pixel Python reference (pulse)')
ax[1].plot(t, ref['Vsf'], 'k', lw=1.6, label='Python ref')
ax[1].plot(cu['t'], cu['Vsf'], 'm--', lw=1.1, label='CUDA array')
ax[1].set_ylabel('Vsf (V)'); ax[1].set_xlabel('time (s)'); ax[1].legend(); ax[1].grid(alpha=.3)
rmse_pr = np.sqrt(np.mean((cu['Vpr']-ref['Vpr'])**2))*1e3
rmse_sf = np.sqrt(np.mean((cu['Vsf']-ref['Vsf'])**2))*1e3
ax[0].text(0.5, 0.05, f'RMSE {rmse_pr:.3f} mV', transform=ax[0].transAxes)
ax[1].text(0.5, 0.05, f'RMSE {rmse_sf:.3f} mV', transform=ax[1].transAxes)
plt.tight_layout(); plt.savefig('cuda_parity.png', dpi=110)
print(f'saved cuda_parity.png  RMSE Vpr={rmse_pr:.3f} mV  Vsf={rmse_sf:.3f} mV')
