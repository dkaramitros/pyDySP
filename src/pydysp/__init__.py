# __init__.py
from .channel import Channel
from .spectra import FourierSpectrum, WelchSpectrum
from .arias import AriasResult
from .response import ResponseSpectrum

__all__ = [
    "Channel",
    "FourierSpectrum",
    "WelchSpectrum",
    "AriasResult",
    "ResponseSpectrum",
]
__version__ = "0.2.0"
