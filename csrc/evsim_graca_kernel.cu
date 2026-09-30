// Graca/Delbruck LPV circuit approximation; input is optical photocurrent (A).
// Physical node voltages are integrated in bounded trapezoidal substeps.
// See include/graca_physics.h and the audit for model/validation limitations.
#include "utils.h"
#include "graca_physics.h"
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>

// Packed state ABI remains 23 planes. Checkpoint semantics changed: the old
// second-order direct-form histories are replaced by physical node voltages.
// Start a new simulator; do not reuse states from the uncorrected implementation.
enum GracaState {
    S_UPR1 = 0,     // previous log-current target (V)
    S_UPR2,         // reserved, zero
    S_YPR1,         // signal Vpr (V)
    S_YPR2,         // signal Vpd (V), NOT previous Vpr
    S_VSF,          // total SF voltage (signal + noise)
    S_VPRSF,        // previous total PR voltage
    S_VSFC,         // SF output driven by signal + PR noise, excluding SF noise
    S_VREF,         // detector reference (V)
    S_TSE,          // time since event (us)
    S_ZMX1, S_ZMX2, S_ZMY1, S_ZMY2, // reserved,reserved, PD noise Vpr,Vpd
    S_ZOX1, S_ZOX2, S_ZOY1, S_ZOY2, // reserved,reserved, PR noise Vpr,Vpd
    S_ZSX1, S_ZSY1, // reserved, SF noise voltage
    S_VSFN, S_VPRSFN, S_MSI, // reserved (unvalidated FPT mode disabled)
    S_IPD_BASE,     // total current used as log reference (A)
    GRACA_NSTATE
};

template <typename scalar_t>
__global__ void evsim_graca_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> image,
    const uint64_t new_time, const uint64_t prev_time,
    torch::PackedTensorAccessor32<double, 3, torch::RestrictPtrTraits> state,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t,
    torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> event_p,
    int32_t* event_count, const graca::Circuit circuit, const double dark_current,
    const double thr_on, const double thr_off, const double refractory_us,
    const bool add_noise, const unsigned long long seed,
    const unsigned long long frame_index, const int n_steps,
    const int max_events, const int height, const int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const double interval_us = static_cast<double>(new_time - prev_time);
    const double step_us = interval_us / n_steps;
    const double dt = step_us * 1e-6;
    const double base = state[S_IPD_BASE][y][x];
    const double next_current = static_cast<double>(image[y][x]) + dark_current;
    const double log_gain = (circuit.ut / circuit.kfb)
                         * graca::operating_point(circuit, next_current).loop_gain;
    double target_old = state[S_UPR1][y][x];
    const double previous_current = base * exp(target_old / log_gain);
    double vpr = state[S_YPR1][y][x], vpd = state[S_YPR2][y][x];
    double vpr_prev = state[S_VPRSF][y][x], vsfc = state[S_VSFC][y][x];
    double vsf = state[S_VSF][y][x], vref = state[S_VREF][y][x];
    double tse = state[S_TSE][y][x];
    double pd_vpr = state[S_ZMY1][y][x], pd_vpd = state[S_ZMY2][y][x];
    double pr_vpr = state[S_ZOY1][y][x], pr_vpd = state[S_ZOY2][y][x];
    double sf_noise = state[S_ZSY1][y][x];

    curandStatePhilox4_32_10_t rng;
    if (add_noise) {
        // Each frame/pixel owns a distinct Philox subsequence; no fixed draw
        // budget can overlap when the frame interval requires more substeps.
        const unsigned long long pixel = static_cast<unsigned long long>(y) * width + x;
        const unsigned long long sequence = frame_index *
            (static_cast<unsigned long long>(width) * height) + pixel;
        curand_init(seed, sequence, 0, &rng);
    }

    for (int step = 1; step <= n_steps; ++step) {
        // Samples define a linear photocurrent trajectory between timestamps.
        // Unrendered flashes cannot be reconstructed by internal integration.
        const double fraction = static_cast<double>(step) / n_steps;
        const double ipd = (1.0 - fraction) * previous_current + fraction * next_current;
        const auto op = graca::operating_point(circuit, ipd);
        const double target_new = log_gain * log(ipd / base);
        graca::signal_step(circuit, op, dt, target_old, target_new, vpd, vpr);
        target_old = target_new;

        if (add_noise) {
            // One-sided PSD = 4*q*I: continuous SDE charge variance = 2*q*I*dt.
            const double pd_charge = sqrt(2.0 * graca::electron_charge * ipd * dt)
                                   * curand_normal_double(&rng);
            const double pr_charge = sqrt(2.0 * graca::electron_charge * circuit.ipr * dt)
                                   * curand_normal_double(&rng);
            graca::node_step(circuit, op, dt, -pd_charge, 0.0, pd_vpd, pd_vpr);
            graca::node_step(circuit, op, dt, 0.0, pr_charge, pr_vpd, pr_vpr);
        }
        const double total_vpr = vpr + pd_vpr + pr_vpr;
        vsfc = graca::sf_step(circuit, dt, vpr_prev, total_vpr, vsfc);
        vpr_prev = total_vpr;
        if (add_noise) {
            const double sf_charge = sqrt(2.0 * graca::electron_charge * circuit.isf * dt)
                                   * curand_normal_double(&rng);
            sf_noise = graca::sf_step(circuit, dt, 0.0, 0.0, sf_noise, sf_charge);
        }
        vsf = vsfc + sf_noise;
        const int polarity = graca::detect_step(vsf, step_us, thr_on, thr_off,
                                                refractory_us, vref, tse);
        if (polarity >= 0) {
            const int index = atomicAdd(event_count, 1);
            if (index < max_events) {
                event_x[index] = static_cast<uint16_t>(x);
                event_y[index] = static_cast<uint16_t>(y);
                // Endpoint sampling: timing error is bounded by a substep for
                // a resolved crossing. Keep uint64 absolute time arithmetic.
                const uint64_t offset = step == n_steps ? new_time - prev_time :
                    static_cast<uint64_t>(llround(step * step_us));
                event_t[index] = prev_time + offset;
                event_p[index] = static_cast<uint8_t>(polarity);
            }
        }
    }
    state[S_UPR1][y][x] = target_old;
    state[S_YPR1][y][x] = vpr;
    state[S_YPR2][y][x] = vpd;
    state[S_VSF][y][x] = vsf;
    state[S_VPRSF][y][x] = vpr_prev;
    state[S_VSFC][y][x] = vsfc;
    state[S_VREF][y][x] = vref;
    state[S_TSE][y][x] = tse;
    state[S_ZMY1][y][x] = pd_vpr; state[S_ZMY2][y][x] = pd_vpd;
    state[S_ZOY1][y][x] = pr_vpr; state[S_ZOY2][y][x] = pr_vpd;
    state[S_ZSY1][y][x] = sf_noise;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_graca(
    const torch::Tensor new_image, const uint64_t new_time, const uint64_t prev_time,
    torch::Tensor state, torch::Tensor event_x_buf, torch::Tensor event_y_buf,
    torch::Tensor event_t_buf, torch::Tensor event_p_buf,
    const double Cpd, const double Cfb, const double Cpr, const double Csf,
    const double Ipr, const double Isf,
    const double kappa_fb, const double kappa_sf, const double VA, const double UT,
    const double full_well_saturation_threshold, const double dark_current,
    const double thr_on, const double thr_off, const double refractory_us,
    const int64_t add_noise, const int64_t stochastic_events,
    const uint64_t seed, const uint64_t frame_index, const int64_t init_steady_state,
    const double dt_us) {
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS(state);
    CHECK_CUDA_CONTIGUOUS(event_x_buf); CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf); CHECK_CUDA_CONTIGUOUS(event_p_buf);
    TORCH_CHECK(new_image.dim() == 2, "new_image must be 2-D (H, W)");
    TORCH_CHECK(new_image.scalar_type() == torch::kFloat32 ||
                new_image.scalar_type() == torch::kFloat64, "image must be float32 or float64 amperes");
    TORCH_CHECK(state.scalar_type() == torch::kFloat64, "Graca circuit state must be float64");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == GRACA_NSTATE &&
                state.size(1) == new_image.size(0) && state.size(2) == new_image.size(1),
                "state must be (23, H, W) matching the image");
    TORCH_CHECK(new_time > prev_time, "new_time must be > prev_time");
    TORCH_CHECK(!stochastic_events, "stochastic_events is disabled: OU bridge is not calibrated or validated");
    for (const double value : {Cpd, Cpr, Csf, Ipr, Isf, kappa_fb, kappa_sf,
                              VA, UT, dark_current, thr_on, thr_off, dt_us}) {
        TORCH_CHECK(std::isfinite(value) && value > 0, "circuit, current, threshold and timestep parameters must be finite and positive");
    }
    TORCH_CHECK(std::isfinite(Cfb) && Cfb >= 0, "Cfb must be finite and nonnegative");
    TORCH_CHECK(kappa_fb <= 1 && kappa_sf <= 1, "kappa factors must be <= 1");
    TORCH_CHECK(std::isfinite(refractory_us) && refractory_us >= 0, "refractory_us must be finite and nonnegative");
    TORCH_CHECK(!std::isnan(full_well_saturation_threshold) &&
                full_well_saturation_threshold > dark_current,
                "current ceiling must exceed dark current, or be positive infinity");
    const auto device = new_image.device();
    for (const auto& tensor : {state, event_x_buf, event_y_buf, event_t_buf, event_p_buf}) {
        TORCH_CHECK(tensor.device() == device, "all Graca tensors must be on the image device");
    }
    for (const auto& buffer : {event_x_buf, event_y_buf, event_t_buf, event_p_buf}) {
        TORCH_CHECK(buffer.dim() == 1 && buffer.numel() == event_x_buf.numel(),
                    "event buffers must be 1-D with identical capacity");
    }
    TORCH_CHECK(event_x_buf.scalar_type() == torch::kUInt16 && event_y_buf.scalar_type() == torch::kUInt16 &&
                event_t_buf.scalar_type() == torch::kUInt64 && event_p_buf.scalar_type() == torch::kUInt8,
                "event buffers must have uint16, uint16, uint64, uint8 dtypes");
    TORCH_CHECK(new_image.size(0) > 0 && new_image.size(0) <= 65535 &&
                new_image.size(1) > 0 && new_image.size(1) <= 65535,
                "image dimensions must be within [1,65535]");
    TORCH_CHECK(state.numel() <= std::numeric_limits<int32_t>::max(), "state exceeds 32-bit accessor range");
    TORCH_CHECK(event_x_buf.numel() > 0 && event_x_buf.numel() <= std::numeric_limits<int32_t>::max(),
                "event capacity must fit a positive int32");
    const c10::cuda::CUDAGuard device_guard(device);
    TORCH_CHECK(torch::isfinite(new_image).all().item<bool>() && new_image.min().item<double>() >= 0,
                "optical photocurrent must be finite, nonnegative amperes");
    TORCH_CHECK(new_image.max().item<double>() + dark_current <= full_well_saturation_threshold,
                "total photocurrent exceeds configured current ceiling; input was not clipped");
    TORCH_CHECK(torch::isfinite(state).all().item<bool>() && state[S_IPD_BASE].min().item<double>() > 0,
                "state must be initialized from the first frame and finite");
    const double steps = std::ceil(static_cast<double>(new_time - prev_time) / dt_us);
    TORCH_CHECK(steps >= 1 && steps <= 1000000, "frame interval requires >1000000 integration substeps; reduce the interval");
    const int n_steps = static_cast<int>(steps);
    TORCH_CHECK(static_cast<double>(new_image.numel()) * n_steps <= std::numeric_limits<int32_t>::max(),
                "maximum possible event count exceeds int32; reduce frame interval or image size");
    const auto pixels = static_cast<uint64_t>(new_image.numel());
    TORCH_CHECK(frame_index <= (std::numeric_limits<uint64_t>::max() - pixels + 1) / pixels,
                "Philox frame subsequence exhausted; reset with a new seed");
    const int height = static_cast<int>(new_image.size(0)), width = static_cast<int>(new_image.size(1));
    const int max_events = static_cast<int>(event_x_buf.numel());
    auto event_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(device));
    const graca::Circuit circuit{Cpd, Cfb, Cpr, Csf, Ipr, Isf, kappa_fb, kappa_sf, VA, UT};
    const dim3 threads(32, 8), blocks(BLOCKS(width, 32), BLOCKS(height, 8));
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_graca_cuda", ([&] {
        evsim_graca_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), new_time, prev_time,
            state.packed_accessor32<double, 3, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(), circuit, dark_current, thr_on, thr_off,
            refractory_us, static_cast<bool>(add_noise), seed, frame_index,
            n_steps, max_events, height, width);
    }));
    const auto cuda_error = cudaGetLastError();
    TORCH_CHECK(cuda_error == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(cuda_error));
    // Reading the count waits on PyTorch's current stream and propagates device errors.
    const int count = event_count.item<int32_t>();
    TORCH_CHECK(count <= max_events, "Graca event buffer overflow: ", count,
                " events exceed capacity ", max_events,
                ". State has advanced; reset and replay with a larger buffer.");
    return std::make_tuple(event_x_buf.slice(0, 0, count), event_y_buf.slice(0, 0, count),
                           event_t_buf.slice(0, 0, count), event_p_buf.slice(0, 0, count));
}
