# arias.py
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AriasResult:
    """Arias intensity time history and significant-duration window.

    Parameters
    ----------
    t : np.ndarray
        Time array corresponding to the input acceleration signal.
    Ia : np.ndarray
        Cumulative Arias intensity evaluated at times ``t``.
    t_start : float
        Start time of the significant-duration window (e.g. 5% point).
    t_end : float
        End time of the significant-duration window (e.g. 95% point).

    Notes
    -----
    The object stores the Husid-style cumulative Arias intensity and
    is primarily used for plotting and extracting the significant-duration
    window for trimming or analysis.
    """

    t: np.ndarray
    Ia: np.ndarray
    t_start: float
    t_end: float

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show_window: bool = True,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """Plot the Arias intensity time history.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        show_window : bool, optional
            If ``True``, vertical dashed lines are drawn at ``t_start`` and
            ``t_end`` to indicate the significant-duration window.
        **plot_kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plotted Arias intensity.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.t, self.Ia, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Arias intensity")
        if show_window:
            ax.axvline(self.t_start, linestyle="--", c="gray")
            ax.axvline(self.t_end, linestyle="--", c="gray")
        ax.grid(True)
        return ax
