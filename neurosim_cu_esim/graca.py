"""Circuit-derived, sampled-current Graca DVS model.

Input is light-generated photocurrent in AMPERES per pixel, excluding dark
current. No normalization or image-dependent rescaling is performed. The kernel
adds dark_current once. Calibration is required for each physical sensor.
"""
import math
import numbers
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple
import torch
from neurosim_cu_esim._backend import evsim_graca_cuda

GRACA_NSTATE = 23
S_TARGET, S_VPR, S_VPD = 0, 2, 3
S_VSF, S_VPRSF, S_VSFC, S_VREF, S_TSE, S_IPD_BASE = 4, 5, 6, 7, 8, 22

class Events(NamedTuple):
    """Coordinates uint16, time in us uint64, polarity ON=1/OFF=0."""
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    p: torch.Tensor

@dataclass
class GracaDVSSimulator:
    """CUDA array model with circuit parameters in SI units.

    forward(photo_A, timestamp_us) supplies instantaneous optical-current
    samples. Currents are linearly interpolated between samples; analog state
    and event detection advance at intervals no longer than dt_us. Rendering
    must itself resolve laser pulses; interpolation cannot recover unseen ones.

    contrast_threshold is an e-fold change of TOTAL photodiode current at
    equilibrium. The finite loop gain and source follower map it to volts.
    full_well_saturation_threshold is a legacy-named optional upper bound
    on total current (A). Exceeding it raises: it is NOT an integrating-pixel
    full-well or a modeled saturation law. None performs no clipping.

    dark_current must be positive. add_noise enables time-resolved shot
    noise; timestep/noise-rate convergence and real-device calibration remain
    required. The previous unvalidated FPT shortcut is explicitly unavailable.
    init_steady_state initializes the deterministic mean only: noise states
    start at zero. Pre-roll an actual laser-off/background sequence until noise
    statistics settle before measuring steady-state background event rates.
    State uses float64 nodal voltages; indices 2/3/4 are Vpr/Vpd/Vsf.
    """
    width: int
    height: int
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
    full_well_saturation_threshold: float | None = None
    dark_current: float = 1.0e-15
    contrast_threshold: float = 0.3
    contrast_threshold_off: float | None = None
    refractory_us: float = 100.0
    dt_us: float = 10.0
    add_noise: bool = False
    stochastic_events: bool = False
    use_first_frame_as_base: bool = True
    init_steady_state: bool = True
    max_events: int | None = None
    seed: int = 0
    device: str | torch.device = 'cuda'
    _state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _event_x_buf: torch.Tensor = field(init=False, repr=False)
    _event_y_buf: torch.Tensor = field(init=False, repr=False)
    _event_t_buf: torch.Tensor = field(init=False, repr=False)
    _event_p_buf: torch.Tensor = field(init=False, repr=False)
    _prev_time: int | None = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False, repr=False)
    _warned_current: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        for name in ('width', 'height'):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, numbers.Integral) or not 1 <= val <= 65535:
                raise ValueError(f'{name} must be an integer in [1, 65535]')
        if self.width * self.height * GRACA_NSTATE >= 2**31:
            raise ValueError('resolution exceeds the 32-bit state accessor capacity')
        for name in ('Cpd', 'Cpr', 'Csf', 'Ipr', 'Isf', 'VA', 'UT', 'dark_current', 'contrast_threshold'):
            self._positive(name, getattr(self, name))
        self._positive('Cfb', self.Cfb, allow_zero=True)
        self._positive('refractory_us', self.refractory_us, allow_zero=True)
        for name in ('kappa_fb', 'kappa_sf'):
            self._positive(name, getattr(self, name))
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be <= 1')
        if self.contrast_threshold_off is not None:
            self._positive('contrast_threshold_off', self.contrast_threshold_off)
        self._positive('dt_us', self.dt_us)
        if self.full_well_saturation_threshold is not None:
            self._positive('full_well_saturation_threshold', self.full_well_saturation_threshold)
            if self.full_well_saturation_threshold <= self.dark_current:
                raise ValueError('current ceiling must exceed dark_current')
        if self.stochastic_events:
            raise NotImplementedError('stochastic_events used an unvalidated FPT heuristic. Use False and time-resolved add_noise with timestep convergence checks.')
        if self.max_events is None:
            self.max_events = self.width * self.height * 16
        if isinstance(self.max_events, bool) or not isinstance(self.max_events, numbers.Integral) or not 1 <= self.max_events < 2**31:
            raise ValueError('max_events must be a positive int32-sized integer')
        self._timestamp(self.seed, name='seed')
        self.device = torch.device(self.device)
        self._init_buffers()

    @staticmethod
    def _positive(name, value, allow_zero=False):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        invalid_sign = value < 0 if allow_zero else value <= 0
        if invalid_sign:
            raise ValueError(f'{name} must be nonnegative' if allow_zero else f'{name} must be positive')

    @staticmethod
    def _timestamp(value, name='timestamp_us'):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value < 2**64:
            raise ValueError(f'{name} must be a nonnegative uint64 integer')
        return int(value)

    @property
    def loop_gain_fraction(self):
        a = self.kappa_fb**2 * self.VA / (2 * self.UT)
        return a / (1 + a)

    @property
    def thr_on(self):
        return self.contrast_threshold * self.kappa_sf * self.UT / self.kappa_fb * self.loop_gain_fraction

    @property
    def thr_off(self):
        tc = self.contrast_threshold if self.contrast_threshold_off is None else self.contrast_threshold_off
        return tc * self.kappa_sf * self.UT / self.kappa_fb * self.loop_gain_fraction

    def _init_buffers(self):
        for name, dtype in (('x', torch.uint16), ('y', torch.uint16), ('t', torch.uint64), ('p', torch.uint8)):
            setattr(self, f'_event_{name}_buf', torch.empty(self.max_events, dtype=dtype, device=self.device))

    def init(self, first_image, timestamp_us=0):
        """Reset and initialize from the FIRST optical-current sample in A."""
        ts = self._timestamp(timestamp_us)
        first_image = self._prepare_image(first_image)
        total = first_image + self.dark_current
        state = torch.zeros((GRACA_NSTATE, self.height, self.width), dtype=torch.float64, device=first_image.device)
        base = total if self.use_first_frame_as_base else torch.full_like(total, self.dark_current)
        target = self.UT / self.kappa_fb * self.loop_gain_fraction * torch.log(total / base)
        state[S_IPD_BASE] = base
        state[S_TARGET] = target
        if self.init_steady_state:
            state[S_VPR] = target
            amp_gain = self.kappa_fb * self.VA / (2 * self.UT)
            state[S_VPD] = -target / amp_gain
            state[S_VPRSF] = target
            for index in (S_VSF, S_VSFC, S_VREF):
                state[index] = self.kappa_sf * target
        state[S_TSE] = self.refractory_us
        self._state = state
        self._prev_time = ts
        self._frame_index = 0

    @property
    def is_initialised(self):
        return self._state is not None

    @property
    def state(self):
        return self._state

    def reset(self, first_image=None, timestamp_us=0):
        self._state = None
        self._prev_time = None
        self._frame_index = 0
        self._warned_current = False
        if first_image is not None:
            self.init(first_image, timestamp_us)

    def forward(self, image, timestamp_us):
        """Advance current samples. Dispatch failures clear possibly partial state."""
        ts = self._timestamp(timestamp_us)
        if not self.is_initialised:
            self.init(image, ts)
            return None
        if ts <= self._prev_time:
            raise ValueError(f'timestamp_us ({ts}) must be > previous timestamp ({self._prev_time})')
        image = self._prepare_image(image)
        ceiling = math.inf if self.full_well_saturation_threshold is None else self.full_well_saturation_threshold
        try:
            x, y, t, p = evsim_graca_cuda(
                image, ts, self._prev_time, self._state,
                self._event_x_buf, self._event_y_buf, self._event_t_buf, self._event_p_buf,
                float(self.Cpd), float(self.Cfb), float(self.Cpr), float(self.Csf),
                float(self.Ipr), float(self.Isf), float(self.kappa_fb), float(self.kappa_sf),
                float(self.VA), float(self.UT), float(ceiling), float(self.dark_current),
                float(self.thr_on), float(self.thr_off), float(self.refractory_us),
                int(bool(self.add_noise)), 0, int(self.seed), int(self._frame_index),
                int(bool(self.init_steady_state)), float(self.dt_us))
        except Exception:
            self.reset()
            raise
        self._prev_time = ts
        self._frame_index += 1
        if x.numel() == 0:
            return None
        return Events(x=x.clone(), y=y.clone(), t=t.clone(), p=p.clone())

    __call__ = forward

    def _prepare_image(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError('Expected torch.Tensor of optical photocurrent in amperes')
        if image.dim() == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
        if image.dim() != 2 or tuple(image.shape) != (self.height, self.width):
            raise ValueError(f'Expected single-channel {(self.height, self.width)} current image, got {tuple(image.shape)}')
        if image.dtype not in (torch.float32, torch.float64):
            raise ValueError('Photocurrent must use a float32 or float64 floating-point dtype in amperes; float16 underflows fA currents')
        image = image.to(device=self.device, dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(image).all()) or bool((image < 0).any()):
            raise ValueError('Optical photocurrent must be finite and nonnegative (amperes)')
        maximum = float(image.max()) + self.dark_current
        if not math.isfinite(maximum):
            raise ValueError('Total photodiode current must be finite')
        if self.full_well_saturation_threshold is not None and maximum > self.full_well_saturation_threshold:
            raise ValueError(f'Total current {maximum:.6g} A exceeds configured current ceiling {self.full_well_saturation_threshold:.6g} A; no silent clipping is performed')
        if maximum >= self.Ipr * 0.1 and not self._warned_current:
            warnings.warn(f'Total photocurrent reaches {maximum:.6g} A versus Ipr={self.Ipr:.6g} A. Check input units and circuit calibration: Ipr >> Ipd no longer holds; this is extrapolation, not measured sensor validation.', RuntimeWarning, stacklevel=2)
            self._warned_current = True
        return image
