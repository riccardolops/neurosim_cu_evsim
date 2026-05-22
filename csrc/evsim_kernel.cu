// neurosim_cu_esim — CUDA kernel for frame-differencing event generation.
//
// Each thread processes one pixel. It computes the log-intensity of the
// incoming grayscale frame, compares it against per-pixel upper/lower
// bounds, and emits an event when the difference exceeds the configured
// contrast threshold. Warp-level ballot/prefix-sum is used to aggregate
// events so that only one atomic per warp is needed.

#include "utils.h"

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void evsim_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t  new_time,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_ub,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_lb,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t,  1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* __restrict__ event_count,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    bool has_event = false;
    bool pos_event = false;

    if (x < width && y < height) {
        const scalar_t cur_log = log(new_image[y][x]);
        const scalar_t ub      = intensity_state_ub[y][x];
        const scalar_t lb      = intensity_state_lb[y][x];

        pos_event = cur_log > ub;
        const bool neg_event = cur_log < lb;
        has_event = pos_event || neg_event;

        // Update state bounds – tighten toward current value when no event,
        // reset around current value when an event fires.
        if (has_event) {
            intensity_state_ub[y][x] = cur_log + static_cast<scalar_t>(contrast_threshold_pos);
            intensity_state_lb[y][x] = cur_log - static_cast<scalar_t>(contrast_threshold_neg);
        } else {
            intensity_state_ub[y][x] = min(ub, cur_log + static_cast<scalar_t>(contrast_threshold_pos));
            intensity_state_lb[y][x] = max(lb, cur_log - static_cast<scalar_t>(contrast_threshold_neg));
        }
    }

    // ------ Warp-level aggregation to minimize atomicAdd contention ------
    const uint32_t tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t   lane_id = tid & 31;

    const uint32_t warp_event_mask  = __ballot_sync(FULL_MASK, has_event);
    const int32_t  warp_event_count = __popc(warp_event_mask);

    if (warp_event_count > 0) {
        int32_t warp_base_idx = 0;
        if (lane_id == 0)
            warp_base_idx = atomicAdd(event_count, warp_event_count);

        // Broadcast base index from lane 0 to all lanes
        warp_base_idx = __shfl_sync(FULL_MASK, warp_base_idx, 0);

        if (has_event) {
            const uint32_t lane_mask          = (1u << lane_id) - 1u;
            const uint32_t preceding_mask     = warp_event_mask & lane_mask;
            const int32_t  thread_event_idx   = __popc(preceding_mask);
            const int32_t  global_event_idx   = warp_base_idx + thread_event_idx;

            if (global_event_idx < static_cast<int32_t>(max_events)) {
                event_x_buf[global_event_idx] = static_cast<uint16_t>(x);
                event_y_buf[global_event_idx] = static_cast<uint16_t>(y);
                event_t_buf[global_event_idx] = new_time;
                event_p_buf[global_event_idx] = pos_event ? 1 : 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-event mode
// ---------------------------------------------------------------------------
//
// Unlike evsim_kernel (which emits at most one event per pixel per frame), this
// kernel emits *as many events as the log-contrast change warrants*: a pixel
// whose log-intensity jumps by N contrast thresholds emits N events.  Their
// timestamps are spread equally across the inter-frame interval
// [prev_time, new_time], with the last event landing exactly on new_time.
//
// Because the per-pixel count is variable, the warp aggregation uses a
// __shfl_up_sync inclusive prefix-sum over per-lane counts (rather than a
// ballot/popc) so each lane learns its contiguous write offset; still only one
// atomicAdd per warp.
template <typename scalar_t>
__global__ void evsim_multi_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t  new_time,
    const uint64_t  prev_time,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_ub,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_lb,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t,  1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* __restrict__ event_count,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t n         = 0;      // number of events this pixel emits this step
    bool    pos_event = false;

    // NOTE: out-of-bounds lanes intentionally do NOT return early — they must
    // still take part in the warp-wide shuffles below (contributing n = 0).
    if (x < width && y < height) {
        const scalar_t cur_log = log(new_image[y][x]);
        const scalar_t ub      = intensity_state_ub[y][x];
        const scalar_t lb      = intensity_state_lb[y][x];

        if (cur_log > ub) {
            // Crossings above the current upper bound (always >= 1 here).
            pos_event = true;
            n = static_cast<int32_t>(
                    floorf(static_cast<float>(cur_log - ub) / contrast_threshold_pos)) + 1;
            if (n > static_cast<int32_t>(max_events))
                n = static_cast<int32_t>(max_events);

            // Discrete-reference update: advance the bounds by exactly n
            // thresholds so the sub-threshold residual is preserved for the
            // next frame (ub_new = ub + n*ct_pos; lb tracks the same ref).
            const scalar_t new_ub =
                ub + static_cast<scalar_t>(n) * static_cast<scalar_t>(contrast_threshold_pos);
            intensity_state_ub[y][x] = new_ub;
            intensity_state_lb[y][x] = new_ub
                - static_cast<scalar_t>(contrast_threshold_pos)
                - static_cast<scalar_t>(contrast_threshold_neg);
        } else if (cur_log < lb) {
            pos_event = false;
            n = static_cast<int32_t>(
                    floorf(static_cast<float>(lb - cur_log) / contrast_threshold_neg)) + 1;
            if (n > static_cast<int32_t>(max_events))
                n = static_cast<int32_t>(max_events);

            const scalar_t new_lb =
                lb - static_cast<scalar_t>(n) * static_cast<scalar_t>(contrast_threshold_neg);
            intensity_state_lb[y][x] = new_lb;
            intensity_state_ub[y][x] = new_lb
                + static_cast<scalar_t>(contrast_threshold_neg)
                + static_cast<scalar_t>(contrast_threshold_pos);
        } else {
            // No event: tighten bounds toward current value (as in single mode).
            intensity_state_ub[y][x] =
                min(ub, cur_log + static_cast<scalar_t>(contrast_threshold_pos));
            intensity_state_lb[y][x] =
                max(lb, cur_log - static_cast<scalar_t>(contrast_threshold_neg));
        }
    }

    // ------ Warp-level prefix-sum to allocate contiguous output slots ------
    const uint32_t tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t   lane_id = tid & 31;

    // Inclusive scan of per-lane counts across the 32 lanes.
    int32_t scan = n;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int32_t v = __shfl_up_sync(FULL_MASK, scan, offset);
        if (lane_id >= offset) scan += v;
    }
    const int32_t warp_total = __shfl_sync(FULL_MASK, scan, 31);

    if (warp_total > 0) {
        int32_t warp_base = 0;
        if (lane_id == 0)
            warp_base = atomicAdd(event_count, warp_total);
        warp_base = __shfl_sync(FULL_MASK, warp_base, 0);

        if (n > 0) {
            const int32_t  my_base = warp_base + (scan - n);  // exclusive prefix
            const uint64_t gap     = new_time - prev_time;
            const uint8_t  pol     = pos_event ? 1 : 0;

            for (int32_t k = 0; k < n; ++k) {
                const int32_t idx = my_base + k;
                if (idx < static_cast<int32_t>(max_events)) {
                    // Equally spaced; the k = n-1 event lands exactly on new_time.
                    const uint64_t t = prev_time
                        + (gap * static_cast<uint64_t>(k + 1)) / static_cast<uint64_t>(n);
                    event_x_buf[idx] = static_cast<uint16_t>(x);
                    event_y_buf[idx] = static_cast<uint16_t>(y);
                    event_t_buf[idx] = t;
                    event_p_buf[idx] = pol;
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim(
    const torch::Tensor new_image,
    const uint64_t new_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos
) {
    // Validate inputs
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_ub);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_lb);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);

    TORCH_CHECK(new_image.dim() == 2,           "new_image must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_ub.dim() == 2,  "intensity_state_ub must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_lb.dim() == 2,  "intensity_state_lb must be 2-D (H, W)");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));

    auto event_count = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(new_image.device()));

    const dim3 threads(32, 32);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_cuda", ([&] {
        evsim_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time,
            intensity_state_ub.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            intensity_state_lb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t,  1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            contrast_threshold_neg,
            contrast_threshold_pos,
            max_events,
            height,
            width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));
    cudaDeviceSynchronize();

    const int32_t num_events =
        std::min(event_count[0].item<int32_t>(),
                 static_cast<int32_t>(max_events));

    if (num_events == 0) {
        auto opts_u16 = torch::dtype(torch::kUInt16).device(new_image.device());
        auto opts_u64 = torch::dtype(torch::kUInt64).device(new_image.device());
        auto opts_u8  = torch::dtype(torch::kUInt8).device(new_image.device());
        return std::make_tuple(
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u64),
            torch::empty({0}, opts_u8)
        );
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events)
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_multi(
    const torch::Tensor new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos
) {
    // Validate inputs
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_ub);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_lb);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);

    TORCH_CHECK(new_image.dim() == 2,           "new_image must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_ub.dim() == 2,  "intensity_state_ub must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_lb.dim() == 2,  "intensity_state_lb must be 2-D (H, W)");
    TORCH_CHECK(new_time >= prev_time,          "new_time must be >= prev_time");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));

    auto event_count = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(new_image.device()));

    const dim3 threads(32, 32);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_multi_cuda", ([&] {
        evsim_multi_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time,
            prev_time,
            intensity_state_ub.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            intensity_state_lb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t,  1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            contrast_threshold_neg,
            contrast_threshold_pos,
            max_events,
            height,
            width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));
    cudaDeviceSynchronize();

    const int32_t num_events =
        std::min(event_count[0].item<int32_t>(),
                 static_cast<int32_t>(max_events));

    if (num_events == 0) {
        auto opts_u16 = torch::dtype(torch::kUInt16).device(new_image.device());
        auto opts_u64 = torch::dtype(torch::kUInt64).device(new_image.device());
        auto opts_u8  = torch::dtype(torch::kUInt8).device(new_image.device());
        return std::make_tuple(
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u64),
            torch::empty({0}, opts_u8)
        );
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events)
    );
}
