#pragma once

#include <torch/extension.h>

// ---- Grid/block sizing ----
#define BLOCKS(N, T) (((N) + (T) - 1) / (T))

// ---- Tensor validation macros ----
#define CHECK_CUDA(x) \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_IS_FLOATING(x)                                          \
    TORCH_CHECK(                                                      \
        (x).scalar_type() == at::ScalarType::Float  ||                \
        (x).scalar_type() == at::ScalarType::Half   ||                \
        (x).scalar_type() == at::ScalarType::Double,                  \
        #x " must be a floating-point tensor (float, half, or double)")

// Combined convenience checks
#define CHECK_CUDA_CONTIGUOUS(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)

#define CHECK_CUDA_CONTIGUOUS_FLOAT(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_IS_FLOATING(x); } while (0)

// ---- Forward declaration of the main dispatch function ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim(
    const torch::Tensor new_image,
    uint64_t new_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    float contrast_threshold_neg,
    float contrast_threshold_pos
);

// ---- Multi-event variant: emits N events per pixel for large log-contrast
//      changes, with timestamps spread equally across [prev_time, new_time]. ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_multi(
    const torch::Tensor new_image,
    uint64_t new_time,
    uint64_t prev_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    float contrast_threshold_neg,
    float contrast_threshold_pos
);

// ---- DVS-Voltmeter stochastic model (ECCV 2022). Linear-intensity input;
//      Brownian-motion-with-drift voltage, Inverse-Gaussian/Levy event times. ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_voltmeter(
    const torch::Tensor new_image,
    uint64_t new_time,
    uint64_t prev_time,
    torch::Tensor base_frame,
    torch::Tensor delta_vd_res,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    double k1, double k2, double k3,
    double k4, double k5, double k6,
    uint64_t seed,
    uint64_t frame_index
);

// ---- Graca & Delbruck (2025) physically-realistic large-signal pixel model.
//      Linear-intensity input -> per-pixel analog front-end (2nd-order PR +
//      1st-order SF) advanced in sub-steps, optional shot noise, v2e events. ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_graca(
    const torch::Tensor new_image,
    uint64_t new_time,
    uint64_t prev_time,
    torch::Tensor state,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    double Cpd, double Cfb, double Cpr, double Csf,
    double Ipr, double Isf,
    double kappa_fb, double kappa_sf, double VA, double UT,
    double full_well_saturation_threshold, double dark_current,
    double thr_on, double thr_off, double refractory_us,
    int64_t add_noise,
    int64_t stochastic_events,
    uint64_t seed,
    uint64_t frame_index,
    int64_t init_steady_state
);
