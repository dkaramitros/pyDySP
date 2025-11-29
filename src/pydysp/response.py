# response.py
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ResponseSpectrum:
    """Elastic response spectrum for a family of SDOF oscillators.

    Parameters
    ----------
    T : np.ndarray
        Natural periods in seconds.
    Sd : np.ndarray
        Spectral displacement values in metres.
    Sv : np.ndarray
        Spectral velocity values in metres per second.
    Sa : np.ndarray
        Spectral pseudo-acceleration values in g.
    ksi : float
        Damping ratio used to compute the spectrum.
    """

    T: np.ndarray
    Sd: np.ndarray
    Sv: np.ndarray
    Sa: np.ndarray
    ksi: float

    def peak(self) -> tuple[float, float]:
        """Return the dominant period and its peak spectral acceleration.

        Returns
        -------
        T_peak : float
            Period at which ``Sa`` is maximum.
        Sa_peak : float
            Maximum spectral acceleration value.

        Raises
        ------
        ValueError
            If the spectral acceleration array ``Sa`` is empty.
        """
        if self.Sa.size == 0:
            raise ValueError(
                "Response spectrum is empty; cannot determine peak period and acceleration"
            )
        idx = int(np.argmax(self.Sa))
        return float(self.T[idx]), float(self.Sa[idx])

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        y: str = "Sa",
        logx: bool = False,
        logy: bool = False,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """Plot one of the response spectra (Sa, Sv or Sd).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        y : {'Sa', 'Sv', 'Sd'}, optional
            Which spectrum to plot: ``Sa`` (default), ``Sv``, or ``Sd``.
        logx : bool, optional
            Use logarithmic x-axis if ``True``.
        logy : bool, optional
            Use logarithmic y-axis if ``True``.
        **plot_kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plotted spectrum.
        """
        if ax is None:
            _, ax = plt.subplots()
        match y:
            case "Sa":
                y_vals = self.Sa
                ylabel = "Spectral acceleration [g]"
            case "Sv":
                y_vals = self.Sv
                ylabel = "Spectral velocity [m/s]"
            case "Sd":
                y_vals = self.Sd
                ylabel = "Spectral displacement [m]"
            case _:
                raise ValueError("y must be one of 'Sa', 'Sv', 'Sd'")
        ax.plot(self.T, y_vals, **plot_kwargs)
        ax.set_xlabel("Period [s]")
        ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both")
        return ax


def sdof_newmark_response(
    acc: np.ndarray,
    dt: float,
    omega: float,
    ksi: float,
) -> tuple[float, float, float]:
    """
    Newmark-beta (average-acceleration) SDOF response to base acceleration.

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history a_g(t) in m/s^2.
    dt : float
        Time step in seconds.
    omega : float
        Circular frequency (rad/s).
    ksi : float
        Damping ratio.

    Returns
    -------
    Sd : float
        Peak relative displacement in metres.
    Sv : float
        Peak relative velocity in metres per second.
    Sa : float
        Peak absolute acceleration in m/s^2.
    """
    n = len(acc)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = 1.0
    k = m * omega**2
    c = 2.0 * ksi * omega * m
    # Newmark average acceleration parameters
    gamma = 0.5
    beta = 0.25
    u = np.zeros(n)
    v = np.zeros(n)
    a_rel = np.zeros(n)
    # Initial relative acceleration from equilibrium
    a_rel[0] = (-acc[0] - c * v[0] - k * u[0]) / m
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 * m + a1 * c
    p = -m * acc
    for i in range(n - 1):
        dp = (
            p[i + 1]
            - p[i]
            + m * (a0 * u[i] + a2 * v[i] + a3 * a_rel[i])
            + c * (a1 * u[i] + a4 * v[i] + a5 * a_rel[i])
        )
        du = dp / k_eff
        u[i + 1] = u[i] + du
        a_rel[i + 1] = a0 * (u[i + 1] - u[i]) - a2 * v[i] - a3 * a_rel[i]
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a_rel[i] + gamma * a_rel[i + 1])
    Sd = float(np.max(np.abs(u)))
    Sv = float(np.max(np.abs(v)))
    a_abs = a_rel + acc
    Sa = float(np.max(np.abs(a_abs)))
    return Sd, Sv, Sa
