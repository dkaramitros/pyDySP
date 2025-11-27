# arias.py
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AriasResult:
    """
    Arias intensity result.

    Attributes
    ----------
    t : np.ndarray
        Time array.
    Ia : np.ndarray
        Arias intensity time history.
    t_start: float
        Time corresponding to the 5% point.
    t_end: float
        Time corresponding to the 95% point.
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
        """
        Plot the Arias intensity time history (Husid plot).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        show_window : bool, optional
            If True, draw vertical lines at t_start and t_end.
        **plot_kwargs
            Extra keyword arguments forwarded to ax.plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
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
