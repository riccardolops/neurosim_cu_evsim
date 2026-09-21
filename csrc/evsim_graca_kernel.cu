// neurosim_cu_esim — CUDA kernel for the physically-realistic large-signal DVS
// pixel model of Graca & Delbruck (2025) / Graca (2024, ETH thesis).
//
// Reference: R. Graca, T. Delbruck, "Towards a physically realistic
// computationally efficient DVS pixel model", arXiv:2505.07386 (2025), and the
// underlying single-pixel MATLAB/Python model in this project.
//
// Each thread owns one pixel and advances its analog front-end across the
// inter-frame interval [prev_time, new_time] in fixed sub-steps of dt_us:
//
//   * Photoreceptor  Zm(s) = Vpr/Ipd   — 2nd-order, operating-point dependent.
//       The large-signal log response is produced by injecting, at the first
//       sub-step of the frame, the "effective log-current increment"
//       A = Ipd_new * ln(Ipd_new / Ipd_old) into Zm and accumulating the output
//       (impulse-response accumulation == step response; validated single-pixel).
//   * Source follower Asf(s) = Vsf/Vpr  — 1st-order, constant coefficients.
//       Run with CONTINUOUS state (the single-pixel bug was resetting it).
//   * Optional shot noise (thesis Eq. 2.45-2.47): three white Gaussian current
//       streams of variance 4*q*I/(2*Ts) filtered by Zm, Zout (-> Vpr) and
//       Zoutsf (-> Vsf); no flicker/RTN.
//   * Change detector: v2e fixed-threshold on Vsf with a refractory period,
//       optionally augmented with stochastic first-passage-time (FPT) event
//       generation (Giraudo & Sacerdote 1999 eq. 3.13; Bibbona et al. 2008 sec. 5).
//       Modeling the Vsf noise as an Ornstein-Uhlenbeck process, the probability
//       that a threshold theta was crossed BETWEEN two sub-steps given the
//       endpoints d_{n-1}, d_n on the same side is
//           P_hidden = exp( -2 (theta - d_{n-1})(theta - d_n) / (s^2 * Ts) ),
//       evaluated as two independent Bernoulli trials (ON / OFF barriers). The OU
//       diffusion s^2*Ts == <(Vsf-noise increment)^2> is calibrated on the fly
//       (mean square of the realized per-sub-step Vsf-noise increment), stored in
//       S_MSI and used on the following frame. This recovers noise event rates at
//       sub-steps far larger than the naive sample-only test (validated single-pixel).
//
// All per-pixel analog state lives in a packed (NSTATE, H, W) tensor so it
// persists across forward() calls. Variable per-pixel event counts are written
// out with the same warp prefix-sum scheme as the other kernels.

#include "utils.h"
#include <curand_kernel.h>

#define FULL_MASK 0xffffffff

// Per-pixel local event buffer (lives in per-thread local memory).
#define GRACA_MAX_EVENTS_PER_PIXEL 16
// Hard cap on sub-steps per frame (defends against huge intervals / tiny dt).
#define GRACA_MAX_SUBSTEPS 8192
// Philox counter budget per pixel per frame (>= draws/substep * substeps).
#define GRACA_DRAWS_PER_FRAME 65536

#define Q_CHARGE 1.602176634e-19

// ---- packed per-pixel state layout (index into dim 0 of the state tensor) ----
enum GracaState {
    S_VPR = 0,      // photoreceptor accumulator (signal only)
    S_DVPR1,        // Zm IIR output history y[n-1]
    S_DVPR2,        // Zm IIR output history y[n-2]
    S_VSF,          // source-follower output (signal+noise) = detector input
    S_VPRSF,        // SF input history x[n-1] (= previous vpr_total fed to SF)
    S_VREF,         // change-detector reference level
    S_TSE,          // microseconds since last event (refractory bookkeeping)
    S_L0,           // previous-frame intensity (operating point memory)
    S_XPR1,         // Zm IIR input history x[n-1] (must persist across frames:
    S_XPR2,         //   with n_sub==1 the b1,b2 feedthrough spans frame steps)
    // ---- noise filter state (used only when add_noise != 0) ----
    S_ZMX1, S_ZMX2, S_ZMY1, S_ZMY2,   // Zm-noise DF-I state (white in, V out)
    S_ZOX1, S_ZOX2, S_ZOY1, S_ZOY2,   // Zout-noise DF-I state
    S_ZSX1, S_ZSY1,                    // Zoutsf-noise DF-I state (1st order)
    // ---- stochastic (first-passage-time) event-generation state ----
    S_VSFN,        // noise-only SF output  Asf{vpr_noise}  (so vsf_noise is separable)
    S_VPRSFN,      // noise-only SF input history (vpr_noise[n-1])
    S_MSI,         // calibrated mean-square Vsf-noise increment <(d noise)^2> = s^2*Ts
    S_VSFC,        // clean SF output Asf{vpr_total} (feedback state; Vsf = vsfc + zsy)
    GRACA_NSTATE
};

// Bilinear (Tustin) transform of a 2nd-order analog TF
//   H(s) = (nb1 s + nb0) / (da2 s^2 + da1 s + 1)
// into normalized digital DF-I coefficients b0,b1,b2 (num) and a1,a2 (den, a0=1).
__device__ __forceinline__ void bilinear2(
    double nb1, double nb0, double da2, double da1, double K0,
    double& b0, double& b1, double& b2, double& a1, double& a2
) {
    const double B0 =  nb1 * K0 + nb0;
    const double B1 =  2.0 * nb0;
    const double B2 = -nb1 * K0 + nb0;
    const double A0 =  da2 * K0 * K0 + da1 * K0 + 1.0;
    const double A1 = -2.0 * da2 * K0 * K0 + 2.0;
    const double A2 =  da2 * K0 * K0 - da1 * K0 + 1.0;
    b0 = B0 / A0; b1 = B1 / A0; b2 = B2 / A0;
    a1 = A1 / A0; a2 = A2 / A0;
}

// Bilinear transform of a 1st-order analog TF  H(s) = nb0 / (da1 s + 1).
__device__ __forceinline__ void bilinear1(
    double nb0, double da1, double K0,
    double& b0, double& b1, double& a1
) {
    const double A0 = da1 * K0 + 1.0;
    b0 = nb0 / A0;
    b1 = nb0 / A0;
    a1 = (1.0 - da1 * K0) / A0;
}

template <typename scalar_t>
__global__ void evsim_graca_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> state,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t,  1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* __restrict__ event_count,
    // model parameters (SI units)
    const double Cpd, const double Cfb, const double Cpr, const double Csf,
    const double Ipr, const double Isf,
    const double kappa_fb, const double kappa_sf, const double VA, const double UT,
    // intensity -> photocurrent mapping:
    //   Ipd = clamp(ipd_max * L/intensity_max, ipd_min, ipd_max)
    const double ipd_max, const double ipd_min, const double intensity_max,
    // change detector
    const double thr_on, const double thr_off, const double refractory_us,
    const double dt_us,
    const int add_noise,
    const int stochastic_events,
    const unsigned long long seed,
    const unsigned long long frame_index,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t n = 0;                                  // events stored by this pixel
    float   loc_t[GRACA_MAX_EVENTS_PER_PIXEL];
    uint8_t loc_p[GRACA_MAX_EVENTS_PER_PIXEL];

    if (x < width && y < height) {
        // ---- operating point from current & previous intensity ----
        // Map linear intensity to photocurrent, clamped to [ipd_min, ipd_max].
        const double Ipd1 = fmin(fmax(ipd_max * ((double)new_image[y][x] / intensity_max),
                                      ipd_min), ipd_max);
        const double Ipd0 = fmin(fmax(ipd_max * ((double)state[S_L0][y][x] / intensity_max),
                                      ipd_min), ipd_max);
        const double Ipd_op = Ipd1;                  // operating point (this frame)
        const double dlog   = log(Ipd1 / Ipd0);      // log-current step

        // ---- sub-step discretization ----
        const double total_dt = (double)(new_time - prev_time);    // us
        int n_sub = (int)llround(total_dt / dt_us);
        if (n_sub < 1) n_sub = 1;
        if (n_sub > GRACA_MAX_SUBSTEPS) n_sub = GRACA_MAX_SUBSTEPS;
        const double dt_act = total_dt / n_sub;       // us, exact span
        const double Ts     = dt_act * 1e-6;          // s
        const double K0     = 2.0 / Ts;               // bilinear 2/Ts

        // ---- small-signal parameters at the operating point ----
        const double gs_fb    = Ipd_op / UT;
        const double gm_fb     = kappa_fb * Ipd_op / UT;
        const double gm_amp_n = kappa_fb * Ipr / UT;
        const double Rout      = VA / (2.0 * Ipr);
        const double Aloop     = gm_amp_n * Rout * gm_fb / gs_fb;
        const double Rin       = 1.0 / gs_fb;

        // shared 2nd-order denominator (Zm and Zout)
        const double da2 = (Rout * (Cpd*Cfb + Cpr*Cfb + Cpr*Cpd)) / (gs_fb * (Aloop + 1.0));
        const double da1 = (Rin * (Cpd + (1.0 + gm_amp_n*Rout)*Cfb)
                            + Rout * (Cpr + (1.0 - gm_fb*Rin)*Cfb)) / (Aloop + 1.0);

        // Zm = Vpr/Ipd
        const double Zm_dc = (1.0/gm_fb) * (Aloop/(Aloop+1.0));
        const double wz_zm = -gm_amp_n / Cfb;
        double zb0, zb1, zb2, za1, za2;
        bilinear2(Zm_dc/wz_zm, Zm_dc, da2, da1, K0, zb0, zb1, zb2, za1, za2);

        // Asf = Vsf/Vpr (1st order, constant)
        const double gs_sf  = Isf / UT;
        const double tau_sf = Csf / gs_sf;
        double sb0, sb1, sa1;
        bilinear1(kappa_sf, tau_sf, K0, sb0, sb1, sa1);

        // ---- noise filter coefficients (only if needed) ----
        double ob0=0,ob1=0,ob2=0,oa1=0,oa2=0;      // Zout
        double nsb0=0,nsb1=0,nsa1=0;                // Zoutsf
        double std_ipd=0, std_ipr=0, std_isf=0;
        curandStatePhilox4_32_10_t rng;
        if (add_noise) {
            const double Zout_dc = Rout / (Aloop + 1.0);
            const double wz_zo   = -gs_fb / (Cpd + Cfb);
            bilinear2(Zout_dc/wz_zo, Zout_dc, da2, da1, K0, ob0, ob1, ob2, oa1, oa2);
            bilinear1(1.0/gs_sf, tau_sf, K0, nsb0, nsb1, nsa1);
            std_ipd = sqrt(4.0*Q_CHARGE*Ipd_op/(2.0*Ts));
            std_ipr = sqrt(4.0*Q_CHARGE*Ipr  /(2.0*Ts));
            std_isf = sqrt(4.0*Q_CHARGE*Isf  /(2.0*Ts));
            const unsigned long long pix = (unsigned long long)y * width + x;
            curand_init(seed, pix, frame_index * GRACA_DRAWS_PER_FRAME, &rng);
        }

        // ---- load persistent state ----
        float Vpr   = state[S_VPR][y][x];
        float dvpr1 = state[S_DVPR1][y][x];
        float dvpr2 = state[S_DVPR2][y][x];
        float Vsf   = state[S_VSF][y][x];
        float vsfc  = state[S_VSFC][y][x];   // clean SF feedback state
        float vprsf = state[S_VPRSF][y][x];
        float Vref  = state[S_VREF][y][x];
        float tse   = state[S_TSE][y][x];
        float xpr1  = state[S_XPR1][y][x];   // PR input history (persisted)
        float xpr2  = state[S_XPR2][y][x];
        
        // FIX: The Zm(s) filter coefficients scale by ~ 1/Ipd_op.
        // If xpr1, xpr2 carry over from a bright frame into a dark frame,
        // they get multiplied by the huge dark gain, causing massive false ON transients.
        // We scale the history by the ratio of operating points to compensate.
        const float op_scale = (float)(Ipd1 / Ipd0);
        xpr1 *= op_scale;
        xpr2 *= op_scale;

        float zmx1=0,zmx2=0,zmy1=0,zmy2=0,zox1=0,zox2=0,zoy1=0,zoy2=0,zsx1=0,zsy1=0;
        if (add_noise) {
            zmx1=state[S_ZMX1][y][x] * op_scale; // Also scale Zm noise history
            zmx2=state[S_ZMX2][y][x] * op_scale;
            zmy1=state[S_ZMY1][y][x]; zmy2=state[S_ZMY2][y][x];
            zox1=state[S_ZOX1][y][x]; zox2=state[S_ZOX2][y][x];
            zoy1=state[S_ZOY1][y][x]; zoy2=state[S_ZOY2][y][x];
            zsx1=state[S_ZSX1][y][x]; zsy1=state[S_ZSY1][y][x];
        }

        // ---- FPT (stochastic event) state ----
        const int use_fpt = (stochastic_events && add_noise);   // needs the noise series
        float vsfn = 0.0f, vprsfn = 0.0f;        // noise-only SF (to isolate vsf_noise)
        float msi_stored = 0.0f;                 // <(d vsf_noise)^2> from previous frame
        double msi_sum = 0.0; int msi_cnt = 0;   // accumulator for this frame's calibration
        float vsf_noise_prev = 0.0f;
        if (use_fpt) {
            vsfn       = state[S_VSFN][y][x];
            vprsfn     = state[S_VPRSFN][y][x];
            msi_stored = state[S_MSI][y][x];
            vsf_noise_prev = vsfn + zsy1;        // last frame's final Vsf noise
        }
        float vsf_prev = Vsf;                     // detector value at previous sub-step

        const float A = (float)(Ipd_op * dlog);          // effective log-current impulse

        // ---- sub-step loop ----
        for (int i = 0; i < n_sub; ++i) {
            const float xin = (i == 0) ? A : 0.0f;       // impulse at frame start

            // photoreceptor Zm (DF-I, 2nd order) -> increment dVpr
            float dvpr = (float)(zb0)*xin + (float)(zb1)*xpr1 + (float)(zb2)*xpr2
                         - (float)(za1)*dvpr1 - (float)(za2)*dvpr2;
            xpr2 = xpr1; xpr1 = xin; dvpr2 = dvpr1; dvpr1 = dvpr;
            Vpr += dvpr;                                  // accumulate (signal)

            // optional shot noise on Vpr (Zm*i_ipd + Zout*i_ipr)
            float vpr_total = Vpr;
            if (add_noise) {
                const float w_ipd = std_ipd * curand_normal(&rng);
                const float w_ipr = std_ipr * curand_normal(&rng);
                float zmy = (float)(zb0)*w_ipd + (float)(zb1)*zmx1 + (float)(zb2)*zmx2
                            - (float)(za1)*zmy1 - (float)(za2)*zmy2;
                zmx2=zmx1; zmx1=w_ipd; zmy2=zmy1; zmy1=zmy;
                float zoy = (float)(ob0)*w_ipr + (float)(ob1)*zox1 + (float)(ob2)*zox2
                            - (float)(oa1)*zoy1 - (float)(oa2)*zoy2;
                zox2=zox1; zox1=w_ipr; zoy2=zoy1; zoy1=zoy;
                vpr_total = Vpr + zmy + zoy;
            }

            // source follower Asf (DF-I, 1st order). The Asf recurrence must feed back
            // its OWN (clean) output vsfc; the Isf shot-noise term zsy is an additive
            // output (Zoutsf has its own filter) and must NOT re-enter the Asf feedback
            // (doing so re-filters zsy through the near-unity SF pole and blows up the
            // Vsf noise). Detector value Vsf = vsfc + zsy.
            const float vsfc_new = (float)(sb0)*vpr_total + (float)(sb1)*vprsf
                                   - (float)(sa1)*vsfc;
            vprsf = vpr_total;
            vsfc = vsfc_new;
            float vsf = vsfc;
            if (add_noise) {
                const float w_isf = std_isf * curand_normal(&rng);
                float zsy = (float)(nsb0)*w_isf + (float)(nsb1)*zsx1 - (float)(nsa1)*zsy1;
                zsx1=w_isf; zsy1=zsy;
                vsf = vsfc + zsy;
                // isolate the Vsf-noise component and calibrate its increment variance
                // (= s^2*Ts for the FPT crossing formula): run a parallel noise-only SF
                // on vpr_noise (= vpr_total - Vpr) and add the Isf contribution zsy.
                if (use_fpt) {
                    const float vpr_noise = vpr_total - Vpr;       // = zmy + zoy
                    const float vsfn_new = (float)(sb0)*vpr_noise + (float)(sb1)*vprsfn
                                           - (float)(sa1)*vsfn;
                    vprsfn = vpr_noise; vsfn = vsfn_new;
                    const float vsf_noise = vsfn + zsy;
                    const float dvn = vsf_noise - vsf_noise_prev;
                    msi_sum += (double)dvn * (double)dvn; msi_cnt++;
                    vsf_noise_prev = vsf_noise;
                }
            }
            Vsf = vsf;

            // change detector with refractory (direct crossing + optional FPT hidden
            // crossing between sub-steps). ON when Vsf-Vref >= +thr_on, OFF when <= -thr_off.
            tse += (float)dt_act;
            if (tse >= (float)refractory_us) {
                const float d  = Vsf      - Vref;        // detector value this sub-step
                const float dp = vsf_prev - Vref;        // and at the previous sub-step
                int   pol  = -1;                         // -1 none, 1 ON, 0 OFF
                float ev_t = 0.0f;
                if (d >= (float)thr_on)        { pol = 1; ev_t = (float)((i + 1) * dt_act); }
                else if (d <= -(float)thr_off) { pol = 0; ev_t = (float)((i + 1) * dt_act); }
                else if (use_fpt && msi_stored > 0.0f) {
                    // two independent Bernoulli trials (upper / lower barrier)
                    if (dp < (float)thr_on) {
                        const float P = expf(-2.0f * ((float)thr_on - dp)
                                             * ((float)thr_on - d) / msi_stored);
                        if (curand_uniform(&rng) < P) { pol = 1; ev_t = (float)((i + 0.5) * dt_act); }
                    }
                    if (pol < 0 && dp > -(float)thr_off) {
                        const float P = expf(-2.0f * (dp + (float)thr_off)
                                             * (d + (float)thr_off) / msi_stored);
                        if (curand_uniform(&rng) < P) { pol = 0; ev_t = (float)((i + 0.5) * dt_act); }
                    }
                }
                if (pol >= 0) {
                    if (n < GRACA_MAX_EVENTS_PER_PIXEL) {
                        loc_t[n] = ev_t; loc_p[n] = (uint8_t)pol; n++;
                    }
                    Vref = Vsf; tse = 0.0f;
                    if (n >= GRACA_MAX_EVENTS_PER_PIXEL) { vsf_prev = Vsf; break; }
                }
            }
            vsf_prev = Vsf;
        }

        // ---- persist state ----
        state[S_VPR][y][x]   = Vpr;
        state[S_DVPR1][y][x] = dvpr1;
        state[S_DVPR2][y][x] = dvpr2;
        state[S_VSF][y][x]   = Vsf;
        state[S_VSFC][y][x]  = vsfc;
        state[S_VPRSF][y][x] = vprsf;
        state[S_VREF][y][x]  = Vref;
        state[S_TSE][y][x]   = tse;
        state[S_L0][y][x]    = new_image[y][x];
        state[S_XPR1][y][x]  = xpr1;
        state[S_XPR2][y][x]  = xpr2;
        if (add_noise) {
            state[S_ZMX1][y][x]=zmx1; state[S_ZMX2][y][x]=zmx2;
            state[S_ZMY1][y][x]=zmy1; state[S_ZMY2][y][x]=zmy2;
            state[S_ZOX1][y][x]=zox1; state[S_ZOX2][y][x]=zox2;
            state[S_ZOY1][y][x]=zoy1; state[S_ZOY2][y][x]=zoy2;
            state[S_ZSX1][y][x]=zsx1; state[S_ZSY1][y][x]=zsy1;
        }
        if (use_fpt) {
            state[S_VSFN][y][x]   = vsfn;
            state[S_VPRSFN][y][x] = vprsfn;
            // calibrated mean-square Vsf-noise increment for next frame's FPT
            state[S_MSI][y][x] = (msi_cnt > 0) ? (float)(msi_sum / msi_cnt) : msi_stored;
        }
    }

    // ------ warp prefix-sum -> contiguous output slots ------
    const uint32_t tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t   lane_id = tid & 31;
    int32_t scan = n;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int32_t v = __shfl_up_sync(FULL_MASK, scan, offset);
        if (lane_id >= offset) scan += v;
    }
    const int32_t warp_total = __shfl_sync(FULL_MASK, scan, 31);

    if (warp_total > 0) {
        int32_t warp_base = 0;
        if (lane_id == 0) warp_base = atomicAdd(event_count, warp_total);
        warp_base = __shfl_sync(FULL_MASK, warp_base, 0);

        if (n > 0) {
            const int32_t my_base = warp_base + (scan - n);
            for (int32_t k = 0; k < n; ++k) {
                const int32_t idx = my_base + k;
                if (idx < (int32_t)max_events) {
                    event_x_buf[idx] = (uint16_t)x;
                    event_y_buf[idx] = (uint16_t)y;
                    event_t_buf[idx] = prev_time + (uint64_t)llroundf(loc_t[k]);
                    event_p_buf[idx] = loc_p[k];
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_graca(
    const torch::Tensor new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::Tensor state,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const double Cpd, const double Cfb, const double Cpr, const double Csf,
    const double Ipr, const double Isf,
    const double kappa_fb, const double kappa_sf, const double VA, const double UT,
    const double ipd_max, const double ipd_min, const double intensity_max,
    const double thr_on, const double thr_off, const double refractory_us,
    const double dt_us,
    const int64_t add_noise,
    const int64_t stochastic_events,
    const uint64_t seed,
    const uint64_t frame_index
) {
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(state);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);

    TORCH_CHECK(new_image.dim() == 2, "new_image must be 2-D (H, W)");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == GRACA_NSTATE,
                "state must be (", (int)GRACA_NSTATE, ", H, W)");
    TORCH_CHECK(new_time > prev_time, "new_time must be > prev_time");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));

    auto event_count = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(new_image.device()));

    // 32-wide blocks keep a warp == one 32-pixel row (needed by the warp
    // prefix-sum). This kernel is register-heavy (double coefficient math +
    // per-thread event/noise state), so use a low block height (256 threads)
    // to stay within the per-block register budget ("too many resources").
    const dim3 threads(32, 8);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_graca_cuda", ([&] {
        evsim_graca_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time, prev_time,
            state.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t,  1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            Cpd, Cfb, Cpr, Csf, Ipr, Isf,
            kappa_fb, kappa_sf, VA, UT,
            ipd_max, ipd_min, intensity_max,
            thr_on, thr_off, refractory_us, dt_us,
            static_cast<int>(add_noise),
            static_cast<int>(stochastic_events),
            static_cast<unsigned long long>(seed),
            static_cast<unsigned long long>(frame_index),
            max_events, height, width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));
    cudaDeviceSynchronize();

    const int32_t num_events =
        std::min(event_count[0].item<int32_t>(), static_cast<int32_t>(max_events));

    if (num_events == 0) {
        auto opts_u16 = torch::dtype(torch::kUInt16).device(new_image.device());
        auto opts_u64 = torch::dtype(torch::kUInt64).device(new_image.device());
        auto opts_u8  = torch::dtype(torch::kUInt8).device(new_image.device());
        return std::make_tuple(
            torch::empty({0}, opts_u16), torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u64), torch::empty({0}, opts_u8));
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events));
}
