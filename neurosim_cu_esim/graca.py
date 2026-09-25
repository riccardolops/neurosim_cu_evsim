"""Physically-realistic large-signal DVS pixel simulator (Graca & Delbruck 2025).

A GPU, array-parallel port of the single-pixel large-signal model from
R. Graca & T. Delbruck, *Towards a physically realistic computationally
efficient DVS pixel model* (arXiv:2505.07386, 2025) and R. Graca, ETH PhD
thesis (2024).

Unlike :class:`~neurosim_cu_esim.simulator.EventSimulator` (log-intensity
threshold crossing) and :class:`~neurosim_cu_esim.voltmeter.DVSVoltmeterSimulator`
(empirical Brownian model), this simulator runs the *physical analog front-end*
of each pixel: a 2nd-order photoreceptor transfer function ``Zm(s)`` and a
1st-order source-follower ``Asf(s)`` derived from circuit analysis, with
small-signal coefficients updated at each frame's operating point (large-signal
linearization). Between frames the difference equations are advanced in fixed
``dt_us`` sub-steps so sub-frame dynamics and event timing are resolved; optional
shot noise is injected per the thesis noise model (Eq. 2.45-2.47).

Input is **linear intensity**, mapped to a per-pixel photocurrent
``Ipd = ipd_max * clamp(L / input_max, eps, 1)`` (the operating point that sets
both the log gain and the time constants).

Default parameters are the project's anchored fit to the paper's Fig. 5 pulse
response (Cpd + 41.7*Cfb = 125 fF decay combination; tau_sf = 1.5 ms).
"""

import logging
from typing import NamedTuple
from dataclasses import dataclass, field

import torch

from neurosim_cu_esim._backend import evsim_graca_cuda

logger = logging.getLogger(__name__)

# Must match the GracaState enum / GRACA_NSTATE in csrc/evsim_graca_kernel.cu.
GRACA_NSTATE = 23
_S_IPD_BASE = 22


class Events(NamedTuple):
    """Container for a batch of events returned by the simulator."""

    x: torch.Tensor
    """Column coordinates (``uint16``)."""
    y: torch.Tensor
    """Row coordinates (``uint16``)."""
    t: torch.Tensor
    """Timestamps in microseconds (``uint64``)."""
    p: torch.Tensor
    """Polarity: 1 = ON (brightness increase), 0 = OFF (``uint8``)."""


@dataclass
class GracaDVSSimulator:
    """CUDA-accelerated physically-realistic large-signal DVS pixel simulator.

    Parameters
    ----------
    width, height : int
        Sensor resolution.
    Cpd, Cfb, Cpr, Csf : float
        Photoreceptor / source-follower capacitances (farads). Defaults are the
        project's anchored fit. NB: the pulse data only constrains
        ``Cpd + 41.7*Cfb`` and ``tau_sf = Csf*UT/Isf`` — the individual splits and
        ``Cpr`` are conventions (see project README identifiability note).
    Ipr, Isf : float
        Photoreceptor and source-follower bias currents (amps).
    kappa_fb, kappa_sf, VA, UT : float
        Subthreshold slope factors, Early voltage (V), thermal voltage (V).
    ipd_max : float
        Photocurrent at full-scale intensity (amps). With the default
        ``1e-12`` the operating point spans ~ipd_min .. ipd_max.
    ipd_min : float
        Photocurrent floor (amps): ``Ipd = clamp(ipd_max*L/input_max, ipd_min,
        ipd_max)``. Avoids ``log(0)`` and sets the dark-current operating point.
    input_max : float
        Full-scale of the input intensity (``1.0`` for ``[0, 1]`` frames,
        ``255.0`` for 8-bit). Used only for the intensity->current mapping.
    contrast_threshold : float
        Event threshold in e-folds of photocurrent (temporal contrast). Mapped
        to a Vsf voltage threshold ``TC * kappa_sf * UT / kappa_fb``.
    contrast_threshold_off : float | None
        Optional separate OFF threshold (defaults to ``contrast_threshold``).
    refractory_us : float
        Per-pixel refractory period (microseconds).
    add_noise : bool
        Inject shot noise (thesis Eq. 2.45-2.47). Default ``False``.
    stochastic_events : bool
        Enable stochastic first-passage-time (FPT) event generation: at each
        sub-step also test, via two Bernoulli trials, whether the Vsf noise
        crossed a threshold *between* sub-steps (Giraudo & Sacerdote 1999;
        Bibbona et al. 2008), recovering noise event rates at large ``dt_us``.
        Requires ``add_noise=True``; ignored otherwise. Default ``False``.
    max_events : int | None
        Cap on events per call (defaults to ``width * height * 16``).
    seed : int
        Philox RNG seed (only relevant when ``add_noise=True``).
    device : str | torch.device
        CUDA device.
    """

    width: int
    height: int
    # --- physical model parameters (SI) — anchored fit defaults ---
    Cpd: float = 71.54e-15
    Cfb: float = 1.00e-15
    Cpr: float = 23.72e-15
    Csf: float = 581.0e-15
    Ipr: float = 3.0e-9
    Isf: float = 10.0e-12
    kappa_fb: float = 0.7
    kappa_sf: float = 0.7
    VA: float = 3.0
    UT: float = 25.8e-3
    # --- intensity -> photocurrent mapping ---
    ipd_max: float = 1.0e-12
    ipd_min: float = 1.0e-15
    input_max: float = 1.0
    # --- change detector ---
    contrast_threshold: float = 0.3
    contrast_threshold_off: float | None = None
    refractory_us: float = 100.0
    # --- noise / misc ---
    add_noise: bool = False
    stochastic_events: bool = False
    use_first_frame_as_base: bool = True
    max_events: int | None = None
    seed: int = 0
    device: str | torch.device = "cuda"

    # ---- internal state ----
    _state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _event_x_buf: torch.Tensor = field(init=False, repr=False)
    _event_y_buf: torch.Tensor = field(init=False, repr=False)
    _event_t_buf: torch.Tensor = field(init=False, repr=False)
    _event_p_buf: torch.Tensor = field(init=False, repr=False)
    _prev_time: int | None = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_events is None:
            self.max_events = self.width * self.height * 16
        self._init_buffers()

    # ------------------------------------------------------------------
    @property
    def thr_on(self) -> float:
        """ON event threshold at Vsf (volts)."""
        return self.contrast_threshold * self.kappa_sf * self.UT / self.kappa_fb

    @property
    def thr_off(self) -> float:
        """OFF event threshold at Vsf (volts)."""
        tc = self.contrast_threshold_off
        if tc is None:
            tc = self.contrast_threshold
        return tc * self.kappa_sf * self.UT / self.kappa_fb

    # ------------------------------------------------------------------
    def _init_buffers(self) -> None:
        assert self.max_events is not None
        dev = self.device
        self._event_x_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_y_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_t_buf = torch.empty(self.max_events, dtype=torch.uint64, device=dev)
        self._event_p_buf = torch.empty(self.max_events, dtype=torch.uint8, device=dev)

    def init(self, first_image: torch.Tensor) -> None:
        """Initialise per-pixel analog state from the first frame."""
        first_image = self._prepare_image(first_image)
        h, w = first_image.shape
        self._state = torch.zeros(
            (GRACA_NSTATE, h, w), dtype=torch.float32, device=first_image.device
        )
        if not self.use_first_frame_as_base:
            self._state[_S_IPD_BASE] = self.ipd_min

    @property
    def is_initialised(self) -> bool:
        return self._state is not None

    def reset(self, first_image: torch.Tensor | None = None) -> None:
        """Clear state and optionally re-initialise from *first_image*."""
        self._state = None
        self._prev_time = None
        self._frame_index = 0
        if first_image is not None:
            self.init(first_image)

    @property
    def state(self) -> torch.Tensor | None:
        """The packed ``(GRACA_NSTATE, H, W)`` analog state tensor."""
        return self._state

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, timestamp_us: int) -> Events | None:
        """Process a new frame (linear intensity) and return generated events.

        The first call initialises state and returns ``None``.
        """
        if not self.is_initialised:
            self.init(image)
            self._prev_time = int(timestamp_us)
            return None

        image = self._prepare_image(image)
        ts = int(timestamp_us)
        prev = self._prev_time if self._prev_time is not None else ts
        if ts <= prev:
            raise ValueError(
                f"timestamp_us ({ts}) must be > previous timestamp ({prev})"
            )

        assert self._state is not None
        x, y, t, p = evsim_graca_cuda(
            image,
            ts,
            prev,
            self._state,
            self._event_x_buf,
            self._event_y_buf,
            self._event_t_buf,
            self._event_p_buf,
            float(self.Cpd), float(self.Cfb), float(self.Cpr), float(self.Csf),
            float(self.Ipr), float(self.Isf),
            float(self.kappa_fb), float(self.kappa_sf), float(self.VA), float(self.UT),
            float(self.ipd_max), float(self.ipd_min), float(self.input_max),
            float(self.thr_on), float(self.thr_off), float(self.refractory_us),
            int(bool(self.add_noise)),
            int(bool(self.stochastic_events)),
            int(self.seed),
            int(self._frame_index),
        )

        self._prev_time = ts
        self._frame_index += 1

        if x.numel() == 0:
            return None
        if x.numel() >= self.max_events:
            logger.warning(
                "Event buffer saturated at max_events=%d; some events were "
                "dropped. Increase max_events.",
                self.max_events,
            )
        return Events(x=x, y=y, t=t, p=p)

    __call__ = forward

    # ------------------------------------------------------------------
    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Move to device, ensure contiguous 2-D float32 tensor."""
        if image.dim() == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
            else:
                raise ValueError(
                    f"Expected single-channel image, got shape {tuple(image.shape)}"
                )
        if image.dim() != 2:
            raise ValueError(f"Expected 2-D (H, W) image, got {image.dim()}-D tensor")
        if not image.is_cuda:
            image = image.to(self.device)
        if image.dtype != torch.float32:
            image = image.float()
        return image.contiguous()
