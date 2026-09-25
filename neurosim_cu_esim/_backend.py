"""Low-level wrapper around the compiled CUDA extension."""

import _neurosim_cu_esim_ext  # type: ignore[import-not-found]
import torch


def evsim_cuda(
    new_image: torch.Tensor,
    new_time: int,
    intensity_state_ub: torch.Tensor,
    intensity_state_lb: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    contrast_threshold_neg: float = 0.35,
    contrast_threshold_pos: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the CUDA kernel that produces events from a single grayscale frame.

    Parameters
    ----------
    new_image : torch.Tensor
        Grayscale frame of shape ``(H, W)`` on a CUDA device.  Values should
        be strictly positive (the kernel computes ``log``).
    new_time : int
        Timestamp in microseconds associated with this frame.
    intensity_state_ub, intensity_state_lb : torch.Tensor
        Per-pixel upper/lower bound state tensors of shape ``(H, W)`` on CUDA.
    event_x_buf, event_y_buf : torch.Tensor
        Pre-allocated ``uint16`` buffers for event *x*/*y* coordinates.
    event_t_buf : torch.Tensor
        Pre-allocated ``uint64`` buffer for event timestamps.
    event_p_buf : torch.Tensor
        Pre-allocated ``uint8`` buffer for event polarities (1 = positive).
    contrast_threshold_neg, contrast_threshold_pos : float
        Thresholds (in log-intensity units) that a pixel must exceed to
        generate a negative/positive event.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x, y, t, p)`` — slices of the pre-allocated buffers trimmed to
        the actual number of events produced.
    """
    return _neurosim_cu_esim_ext.evsim(
        new_image,
        new_time,
        intensity_state_ub,
        intensity_state_lb,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        contrast_threshold_neg,
        contrast_threshold_pos,
    )


def evsim_multi_cuda(
    new_image: torch.Tensor,
    new_time: int,
    prev_time: int,
    intensity_state_ub: torch.Tensor,
    intensity_state_lb: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    contrast_threshold_neg: float = 0.35,
    contrast_threshold_pos: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-event variant of :func:`evsim_cuda`.

    Emits *multiple* events per pixel when the log-intensity change spans more
    than one contrast threshold, instead of capping at one event per pixel.
    The ``n`` events generated for a pixel are assigned timestamps spread
    equally across the inter-frame interval, so that ``t_k = prev_time +
    (new_time - prev_time) * (k + 1) / n`` for ``k = 0 .. n-1`` — the last
    event lands exactly on ``new_time``.

    Parameters
    ----------
    prev_time : int
        Timestamp (microseconds) of the *previous* frame; defines the lower end
        of the interval that emitted events are spread across.  Must satisfy
        ``prev_time <= new_time``.

    All other parameters match :func:`evsim_cuda`.
    """
    return _neurosim_cu_esim_ext.evsim_multi(
        new_image,
        new_time,
        prev_time,
        intensity_state_ub,
        intensity_state_lb,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        contrast_threshold_neg,
        contrast_threshold_pos,
    )


def evsim_voltmeter_cuda(
    new_image: torch.Tensor,
    new_time: int,
    prev_time: int,
    base_frame: torch.Tensor,
    delta_vd_res: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    k6: float,
    seed: int,
    frame_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the DVS-Voltmeter stochastic event kernel for one frame step.

    Models each pixel's sensor voltage as a Brownian motion with drift derived
    from the two frames (paper Eq. 10/11) and emits events at threshold
    crossings, with timestamps sampled from the Inverse-Gaussian / Levy
    first-passage-time distribution.

    Parameters
    ----------
    new_image : torch.Tensor
        Grayscale ``(H, W)`` frame on CUDA, **linear intensity** (e.g. 0-255,
        the scale the ``k`` params are calibrated to). *Not* log-intensity.
    new_time, prev_time : int
        Current / previous frame timestamps in microseconds (``new_time >
        prev_time``).
    base_frame : torch.Tensor
        Per-pixel previous intensity ``L0`` state ``(H, W)``; updated in place.
    delta_vd_res : torch.Tensor
        Per-pixel residual voltage state ``(H, W)``; updated in place.
    k1..k6 : float
        DVS-Voltmeter model parameters (camera-specific calibration).
    seed, frame_index : int
        Philox RNG seed and per-frame counter offset (for reproducibility and
        fresh randomness across frames).

    Returns
    -------
    tuple of ``(x, y, t, p)`` slices of the pre-allocated buffers.
    """
    return _neurosim_cu_esim_ext.evsim_voltmeter(
        new_image,
        new_time,
        prev_time,
        base_frame,
        delta_vd_res,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        seed,
        frame_index,
    )


def evsim_graca_cuda(
    new_image: torch.Tensor,
    new_time: int,
    prev_time: int,
    state: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    Cpd: float,
    Cfb: float,
    Cpr: float,
    Csf: float,
    Ipr: float,
    Isf: float,
    kappa_fb: float,
    kappa_sf: float,
    VA: float,
    UT: float,
    ipd_max: float,
    ipd_min: float,
    intensity_max: float,
    thr_on: float,
    thr_off: float,
    refractory_us: float,
    add_noise: int,
    stochastic_events: int,
    seed: int,
    frame_index: int,
    init_steady_state: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the Graca & Delbruck large-signal physical pixel kernel for one step.

    Each pixel runs a 2nd-order photoreceptor + 1st-order source-follower
    difference equation, advanced in ``dt_us`` sub-steps across
    ``[prev_time, new_time]``, with operating-point-dependent coefficients,
    optional shot noise, and a v2e fixed-threshold change detector.

    Parameters
    ----------
    new_image : torch.Tensor
        Grayscale ``(H, W)`` frame on CUDA, **linear intensity** (mapped to a
        per-pixel photocurrent ``Ipd = ipd_max * clamp(L/intensity_max, eps, 1)``).
    state : torch.Tensor
        Packed per-pixel analog state ``(GRACA_NSTATE, H, W)`` float32; updated
        in place. ``GRACA_NSTATE == 24``.
    Cpd, Cfb, Cpr, Csf : float
        Photoreceptor / source-follower capacitances (F).
    Ipr, Isf : float
        Photoreceptor and source-follower bias currents (A).
    kappa_fb, kappa_sf, VA, UT : float
        Subthreshold slopes, Early voltage (V), thermal voltage (V).
    ipd_max, intensity_max : float
        Intensity-to-photocurrent mapping (max photocurrent A; input full-scale).
    thr_on, thr_off : float
        Event thresholds at Vsf (volts).
    refractory_us, dt_us : float
        Refractory period (microseconds).
    add_noise : int
        ``1`` to add shot noise, ``0`` for the deterministic signal model.
    stochastic_events : int
        ``1`` to enable stochastic first-passage-time event generation (noise
        crossings between sub-steps; requires ``add_noise=1``), ``0`` for the
        plain v2e fixed-threshold detector.
    seed, frame_index : int
        Philox RNG seed and per-frame counter offset.

    Returns
    -------
    tuple of ``(x, y, t, p)`` slices of the pre-allocated buffers.
    """
    return _neurosim_cu_esim_ext.evsim_graca(
        new_image,
        new_time,
        prev_time,
        state,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        Cpd,
        Cfb,
        Cpr,
        Csf,
        Ipr,
        Isf,
        kappa_fb,
        kappa_sf,
        VA,
        UT,
        ipd_max,
        ipd_min,
        intensity_max,
        thr_on,
        thr_off,
        refractory_us,
        add_noise,
        stochastic_events,
        seed,
        frame_index,
        init_steady_state,
    )
