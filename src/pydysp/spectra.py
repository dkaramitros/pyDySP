# spectra.py
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FourierSpectrum:
    """
    Single-sided Fourier amplitude spectrum.

    Attributes
    ----------
    f : np.ndarray
        Frequency array (Hz).
    s : np.ndarray
        Amplitude spectrum |FFT|.
    """

    f: np.ndarray
    s: np.ndarray

    def peak(self) -> tuple[float, float]:
        """
        Return the frequency and amplitude at the maximum spectral peak.

        Returns
        -------
        f_peak : float
            Frequency at which the spectrum is maximum.
        s_peak : float
            Maximum amplitude.
        """
        if self.s.size == 0:
            raise ValueError("Empty spectrum has no peak")
        idx = int(np.argmax(self.s))
        return float(self.f[idx]), float(self.s[idx])

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        fmax: Optional[float] = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Fourier amplitude spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        fmax : float, optional
            Upper x-limit for frequency axis (Default 50.0).
        **plot_kwargs
            Extra keyword arguments forwarded to ax.plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.f, self.s, **plot_kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Fourier amplitude")
        if fmax is not None:
            ax.set_xlim(0.0, fmax)
        ax.grid(True)
        return ax


@dataclass
class WelchSpectrum:
    """
    Power spectral density (PSD) from Welch's method.

    Attributes
    ----------
    f : np.ndarray
        Frequency array (Hz).
    p : np.ndarray
        PSD values corresponding to f.
    """

    f: np.ndarray
    p: np.ndarray

    def peak(self) -> tuple[float, float]:
        """
        Return the frequency and PSD value at the maximum spectral peak.

        Returns
        -------
        f_peak : float
            Frequency at which the PSD is maximum.
        p_peak : float
            Maximum PSD value.
        """
        if self.p.size == 0:
            raise ValueError("Empty PSD has no peak")
        idx = int(np.argmax(self.p))
        return float(self.f[idx]), float(self.p[idx])

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        fmax: Optional[float] = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Welch power spectral density.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        fmax : float, optional
            Upper x-limit for frequency axis (Default 50.0).
        **plot_kwargs
            Extra keyword arguments forwarded to ax.plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.f, self.p, **plot_kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")
        if fmax is not None:
            ax.set_xlim(0.0, fmax)
        ax.grid(True)
        return ax
