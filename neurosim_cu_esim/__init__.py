"""neurosim_cu_esim — CUDA-accelerated frame-differencing event simulator."""

from importlib.metadata import PackageNotFoundError, version

from neurosim_cu_esim.simulator import EventSimulator
from neurosim_cu_esim.voltmeter import DVSVoltmeterSimulator
from neurosim_cu_esim.graca import GracaDVSSimulator

try:
    __version__ = version("neurosim_cu_esim")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "EventSimulator",
    "DVSVoltmeterSimulator",
    "GracaDVSSimulator",
    "__version__",
]
