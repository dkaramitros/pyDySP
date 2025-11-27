from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.integrate import cumulative_trapezoid


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
        fmax: Optional[float] = 50,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Fourier amplitude spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        fmax : float, optional
            Upper x-limit for frequency axis (lower is fixed at 0).
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
        fmax: Optional[float] = 50,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Welch power spectral density.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        fmax : float, optional
            Upper x-limit for frequency axis (lower is fixed at 0).
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


@dataclass
class ResponseSpectrum:
    """
    Elastic response spectrum for an SDOF oscillator family.

    Attributes
    ----------
    T : np.ndarray
        Natural periods (s).
    Sd : np.ndarray
        Displacement spectrum for each period.
    Sv : np.ndarray
        Velocity spectrum for each period.
    Sa : np.ndarray
        Pseudo-acceleration spectrum for each period.
    ksi : float
        Damping ratio used for the spectrum.
    """

    T: np.ndarray
    Sd: np.ndarray
    Sv: np.ndarray
    Sa: np.ndarray
    ksi: float

    def peak(self) -> tuple[float, float]:
        """
        Return the dominant period and its corresponding peak spectral acceleration.

        Returns
        -------
        T_peak : float
            Period at which Sa is maximum.
        Sa_peak : float
            Maximum spectral acceleration value.
        """
        if self.Sa.size == 0:
            raise ValueError("Empty response spectrum has no peak")
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
        """
        Plot the response spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        y : {'Sa', 'Sv', 'Sd'}, optional
            Which spectrum to plot: Sa (default), Sv, or Sd.
        logx : bool, optional
            Use logarithmic x-axis if True.
        logy : bool, optional
            Use logarithmic y-axis if True.
        **plot_kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        match y:
            case "Sa":
                y = self.Sa
                ylabel = "Spectral acceleration"
            case "Sv":
                y = self.Sv
                ylabel = "Spectral velocity"
            case "Sd":
                y = self.Sd
                ylabel = "Spectral displacement"
            case _:
                raise ValueError("y must be one of 'Sa', 'Sv', 'Sd'")
        ax.plot(self.T, y, **plot_kwargs)
        ax.set_xlabel("Period [s]")
        ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both")
        return ax


@dataclass
class Channel:
    """
    Represents a single experimental time-history channel.

    Features
    --------
    - Stores a 1D signal (`data`) and its time axis (`time`, or `dt` and `t0`).
    - Keeps naming information for plotting and reports, as well as flexible tags and free-form metadata.
    - Maintains processing parameters (e.g. for trimming and filtering) without modifying the raw data.
    """

    # Default processing parameters
    DRIFT_DEFAULTS = {"points": 50}
    FILTER_DEFAULTS = {"btype": "lowpass", "fc": 50.0, "order": 2}
    BASELINE_DEFAULTS = {"type": "linear"}

    # 1D numeric signal values (raw data)
    data: np.ndarray
    # Sampling interval in seconds (used if explicit `time` is not provided)
    dt: Optional[float] = None
    # Start time in seconds (used when constructing `time` from `dt`)
    t0: float = 0.0
    # Explicit time array (same length/shape as `data`, if provided)
    time: Optional[np.ndarray] = None

    # Original name as found in the input file
    name_input: Optional[str] = None
    # Your preferred short name, e.g. "Acc1"
    name_user: Optional[str] = None
    # Axis label for plotting, e.g. "Acc1: Footing (g)"
    label_axis: Optional[str] = None
    # Legend label for plotting, e.g. "Acc1: Footing"
    label_legend: Optional[str] = None
    # Longer human-readable description of the channel, e.g. "Acceleration at footing level"
    description_long: Optional[str] = None
    # Physical quantity type, e.g. "displacement", "force", "acceleration"
    quantity: Optional[str] = None
    # Engineering units, e.g. "m", "kN", "g"
    units: Optional[str] = None
    # Raw DAQ units, e.g. "V"
    raw_units: Optional[str] = None
    # Multiplicative factor to convert raw data to physical units (e.g. g/V)
    calibration_factor: float = 1.0

    # Drift correction parameters: { "points": 50 }
    drift_params: Dict[str, Any] = field(default_factory=dict)
    # Filtering parameters, e.g. { "btype": "lowpass", "fc": 50.0, "order": 2 }
    filter_params: Dict[str, Any] = field(default_factory=dict)
    # Baseline correction parameters, e.g. { "type": "linear" }
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    # Trimming parameters: {"t_start":, "t_end":} in seconds
    trim_params: Dict[str, Any] = field(default_factory=dict)
    # Free-form notes about processing steps (trimming, filtering, calibration, etc.)
    processing_notes: list[str] = field(default_factory=list)

    # Tags used for grouping
    tags: set[str] = field(default_factory=set)
    # Free-form metadata, e.g. sensor type
    meta: Dict[str, Any] = field(default_factory=dict)

    # Internal cache for processed data (not part of the public API)
    _cache_processed_time: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _cache_processed_data: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _cache_signature: Optional[tuple] = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Normalise arrays, infer missing timing info, and set sensible defaults
        for names / labels / tags after dataclass initialisation.
        """
        # Ensure data is a 1D NumPy array
        self.data = np.asarray(self.data)
        if self.data.ndim != 1:
            raise ValueError("Channel.data must be a 1D array (single time history)")

        # Handle time / dt relationship
        if self.time is not None:
            self.time = np.asarray(self.time)
            if self.time.shape != self.data.shape:
                raise ValueError("Time and data must have the same shape")
            if self.dt is None and len(self.time) > 1:
                self.dt = float(self.time[1] - self.time[0])
            if self.t0 == 0.0 and len(self.time) > 0:
                self.t0 = float(self.time[0])
        else:
            if self.dt is None:
                raise ValueError("Either 'time' or 'dt' must be provided")
            if self.dt <= 0:
                raise ValueError("dt must be positive")
            n = self.data.shape[0]
            self.time = self.t0 + self.dt * np.arange(n)

        # Normalise and auto-add tags
        self.tags = set(self.tags)
        if self.quantity is not None:
            self.tags.add(f"q:{self.quantity}")

        # Sensible fallbacks for names and labels
        if self.name_user is None and self.name_input is not None:
            self.name_user = self.name_input
        if self.label_legend is None and self.name_user is not None:
            self.label_legend = self.name_user
        if self.label_axis is None and self.units is not None:
            base = self.label_legend or self.name_user or ""
            units = f"[{self.units}]" or ""
            self.label_axis = f"{base} {units}".strip()

    # ------------------------------------------------------------------ #
    # Internal cache management
    # ------------------------------------------------------------------ #

    def _clear_cache(self) -> None:
        """
        Clear cached processed data (called when processing parameters change).
        """
        self._cache_processed_time = None
        self._cache_processed_data = None
        self._cache_signature = None

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def info(self) -> str:
        """
        Return a clean, human-readable summary of channel metadata.
        """
        lines = []
        # Header
        title = self.name_user or self.name_input or "(unnamed channel)"
        lines.append(f"Channel: {title}")
        lines.append("-" * (len(title) + 9))
        # Basic signal info
        lines.append(f"Length: {len(self.data)} samples")
        if self.dt is not None and self.dt > 0:
            lines.append(f"Sampling frequency: fs = {1.0 / self.dt:g} Hz")
            lines.append(f"Timestep: dt = {self.dt:g} s")
        else:
            lines.append("Sampling frequency: fs = (unknown)")
            lines.append("Timestep: dt = (unknown)")
        lines.append(f"Start time: t0 = {self.t0} s")
        # Physical meaning & calibration
        if any(
            [self.quantity, self.units, self.raw_units, self.calibration_factor != 1.0]
        ):
            lines.append("\nPhysical meaning & calibration:")
            if self.quantity:
                lines.append(f"  Quantity: {self.quantity}")
            if self.units:
                lines.append(f"  Units: {self.units}")
            if self.raw_units:
                lines.append(f"  Raw units: {self.raw_units}")
            if self.calibration_factor != 1.0:
                lines.append(
                    f"  Calibration factor: {self.calibration_factor}"
                    + f" {self.units or 'units'}/{self.raw_units or 'raw'}"
                )
        # Naming / labels
        if any(
            [
                self.name_input,
                self.name_user,
                self.label_legend,
                self.label_axis,
                self.description_long,
            ]
        ):
            lines.append("\nNaming / labels:")
            if self.name_input:
                lines.append(f"  Input name: {self.name_input}")
            if self.name_user:
                lines.append(f"  User name: {self.name_user}")
            if self.label_legend:
                lines.append(f"  Legend label: {self.label_legend}")
            if self.label_axis:
                lines.append(f"  Axis label: {self.label_axis}")
            if self.description_long:
                lines.append(f"  Description: {self.description_long}")
        # Tags
        if self.tags:
            lines.append("\nTags:")
            taglist = "\n  ".join(sorted(self.tags))
            lines.append(f"  {taglist}")
        # Processing
        if (
            self.drift_params
            or self.filter_params
            or self.baseline_params
            or self.trim_params
            or self.processing_notes
        ):
            lines.append("\nProcessing:")
            if self.drift_params:
                lines.append(f"  Drift params: {self.drift_params}")
            if self.filter_params:
                lines.append(f"  Filter params: {self.filter_params}")
            if self.baseline_params:
                lines.append(f"  Baseline params: {self.baseline_params}")
            if self.trim_params:
                lines.append(f"  Trim params: {self.trim_params}")
            if self.processing_notes:
                lines.append("  Notes:")
                for note in self.processing_notes:
                    lines.append(f"    - {note}")
        # Free-form metadata
        if self.meta:
            lines.append("\nMetadata:")
            for k, v in self.meta.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Processing methods that create new Channels
    # ------------------------------------------------------------------ #

    def drift_corrected(self, **override: Any) -> "Channel":
        """
        Return a new Channel with updated drift parameters.

        Actual correction is applied in processed().
        Defaults: {"points": 50}
        """
        params = {**self.DRIFT_DEFAULTS, **self.drift_params, **override}
        new = replace(self, drift_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"drift_corrected"})
        new.processing_notes = [
            *self.processing_notes,
            f"Drift params set: {params}",
        ]
        return new

    def filtered(self, **override: Any) -> "Channel":
        """
        Return a new Channel with updated Butterworth filter parameters.

        Actual filtering is applied in processed().
        Defaults: {"btype": "lowpass", "fc": 50.0, "order": 2}
        """
        params = {**self.FILTER_DEFAULTS, **self.filter_params, **override}
        new = replace(self, filter_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"filtered"})
        new.processing_notes = [
            *self.processing_notes,
            f"Filter params set: {params}",
        ]
        return new

    def baseline_corrected(self, **override: Any) -> "Channel":
        """
        Return a new Channel with updated baseline correction parameters.

        Actual detrending is applied in processed().
        Defaults: {"type": "linear"}
        """
        params = {**self.BASELINE_DEFAULTS, **self.baseline_params, **override}
        new = replace(self, baseline_params=params)
        new._clear_cache()
        arg_str = ", ".join(f"{k}={v}" for k, v in params.items()) or ""
        new.tags = set(self.tags).union({"baseline_corrected"})
        new.processing_notes = [
            *self.processing_notes,
            f"Baseline params set: detrend({arg_str})",
        ]
        return new

    def trimmed(self, **override: Any) -> "Channel":
        """
        Return a new Channel with updated trim window parameters.

        Actual trimming is applied in processed().
        """
        defaults = {
            "t_start": float(self.time[0]),
            "t_end": float(self.time[-1]),
        }
        params = {**defaults, **self.trim_params, **override}
        t_start = params["t_start"]
        t_end = params["t_end"]
        new = replace(self, trim_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"trimmed"})
        new.processing_notes = [
            *self.processing_notes,
            f"Trim params set: {t_start}–{t_end} s",
        ]
        return new

    # ------------------------------------------------------------------ #
    # Trimmers
    # ------------------------------------------------------------------ #

    def trim_by_threshold(
        self,
        threshold: float = 0.01,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Return a new Channel trimmed to where the signal exceeds a threshold.

        This is the classic 'bracketed duration' style trimming. The window is
        defined from the first to the last sample where the signal exceeds the
        threshold, optionally in absolute value, with optional time buffers.

        Parameters
        ----------
        threshold : float
            Amplitude threshold in signal units (e.g. 0.01 if data is in g).
        use_abs : bool, optional
            If True (default), use |y| >= threshold. If False, use y >= threshold.
        buffer_before : float, optional
            Extra time (s) added before the first exceedance. Default 0.0.
        buffer_after : float, optional
            Extra time (s) added after the last exceedance. Default 0.0.
        processed : bool, optional
            If True (default), use the processed signal.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        Channel
            New Channel with updated trim_params.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal")
        if use_abs:
            mask = np.abs(y) >= threshold
        else:
            mask = y >= threshold
        if not np.any(mask):
            raise ValueError("No samples exceed the specified threshold")
        i_start = int(np.argmax(mask))
        i_end = int(len(mask) - 1 - np.argmax(mask[::-1]))
        t_start = float(t[i_start]) - buffer_before
        t_end = float(t[i_end]) + buffer_after
        # Clamp to original time range
        t_start = max(t_start, float(t[0]))
        t_end = min(t_end, float(t[-1]))
        if t_end <= t_start:
            raise ValueError("Computed trim window is empty after buffering")
        return self.trimmed(t_start=t_start, t_end=t_end)

    def trim_by_fraction_of_peak(
        self,
        fraction: float = 0.05,
        *,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Return a new Channel trimmed to where the signal is above a fraction
        of its peak amplitude.

        Parameters
        ----------
        fraction : float
            Fraction of the peak amplitude in (0, 1], e.g. 0.05 for 5% of peak.
        use_abs : bool, optional
            If True (default), peak and threshold are based on |y|.
            If False, peak and threshold are based on y (positive peaks only).
        buffer_before : float, optional
            Extra time (s) added before the first exceedance. Default 0.0.
        buffer_after : float, optional
            Extra time (s) added after the last exceedance. Default 0.0.
        processed : bool, optional
            If True (default), use the processed signal.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        Channel
            New Channel with updated trim_params.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")
        _, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal")
        if use_abs:
            peak = float(np.max(np.abs(y)))
        else:
            peak = float(np.max(y))
        if peak <= 0.0:
            raise ValueError(
                "Signal peak is non-positive; cannot define threshold from peak"
            )
        threshold = fraction * peak
        return self.trim_by_threshold(
            threshold=threshold,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )

    def trim_by_arias(
        self,
        lower: float = 0.05,
        upper: float = 0.95,
        g: float = 9.81,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Return a new Channel trimmed to the Arias-intensity-based significant
        duration window.

        By default this uses the classic 5%–95% Arias intensity window.

        Parameters
        ----------
        lower : float, optional
            Lower fraction of final Arias intensity, in [0, 1). Default 0.05.
        upper : float, optional
            Upper fraction of final Arias intensity, in (0, 1]. Default 0.95.
        g : float, optional
            Acceleration due to gravity (m/s^2). Default 9.81.
        buffer_before : float, optional
            Extra time (s) added before the lower Arias crossing. Default 0.0.
        buffer_after : float, optional
            Extra time (s) added after the upper Arias crossing. Default 0.0.
        processed : bool, optional
            If True (default), use the processed signal.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        Channel
            New Channel with updated trim_params.
        """
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("Require 0 <= lower < upper <= 1")
        res = self.arias_intensity(g=g, processed=processed, use_cache=use_cache)
        t = res.t
        Ia = res.Ia
        if Ia.size == 0:
            raise ValueError("Cannot trim by Arias intensity for empty signal")
        Ia_final = float(Ia[-1])
        if Ia_final <= 0.0:
            raise ValueError("Final Arias intensity is non-positive")
        # Find indices where cumulative Arias crosses the requested fractions
        idx_lower = int(np.argmax(Ia >= lower * Ia_final))
        idx_upper = int(np.argmax(Ia >= upper * Ia_final))
        t_lower = float(t[idx_lower]) - buffer_before
        t_upper = float(t[idx_upper]) + buffer_after
        # Clamp to original time range
        t_lower = max(t_lower, float(t[0]))
        t_upper = min(t_upper, float(t[-1]))
        if t_upper <= t_lower:
            raise ValueError("Computed Arias trim window is empty after buffering")
        return self.trimmed(t_start=t_lower, t_end=t_upper)

    # ------------------------------------------------------------------ #
    # Processed
    # ------------------------------------------------------------------ #

    def processed(self, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time, data) with processing applied in the following order:

        1. Drift correction via drift_params
        2. Butterworth filter via filter_params
        3. Baseline correction via baseline_params
        4. Trimming via trim_params
        5. Calibration via calibration_factor
        """
        signature = (
            tuple(sorted(self.drift_params.items())),
            tuple(sorted(self.filter_params.items())),
            tuple(sorted(self.baseline_params.items())),
            tuple(sorted(self.trim_params.items())),
            self.calibration_factor,
        )
        if (
            use_cache
            and self._cache_signature == signature
            and self._cache_processed_data is not None
            and self._cache_processed_time is not None
        ):
            return self._cache_processed_time, self._cache_processed_data
        # Start from raw
        t = self.time
        y = self.data.astype(float, copy=False)
        # 1. Drift correction
        if self.drift_params:
            params = {**self.DRIFT_DEFAULTS, **self.drift_params}
            points = params["points"]
            if points > len(y):
                raise ValueError(
                    "Number of points for drift correction exceeds data length"
                )
            drift_value = float(np.mean(y[:points]))
            y = y - drift_value
        # 2. Filtering
        if self.filter_params:
            fs = 1.0 / self.dt
            params = {**self.FILTER_DEFAULTS, **self.filter_params}
            btype = params["btype"]
            order = params["order"]
            if btype in ("lowpass", "highpass"):
                fc = params["fc"]
                Wn = 2 * fc / fs
            elif btype in ("bandpass", "bandstop"):
                f1 = params.get("f1")
                f2 = params.get("f2")
                if f1 is None or f2 is None:
                    raise ValueError("Band filters require f1 and f2")
                Wn = [2 * f1 / fs, 2 * f2 / fs]
            else:
                raise ValueError(f"Unsupported filter mode: {btype!r}")
            if isinstance(Wn, (list, tuple)):
                if not (0 < Wn[0] < Wn[1] < 1):
                    raise ValueError(
                        "Normalized band edges must satisfy 0 < f1 < f2 < fs/2; "
                        f"got Wn={Wn} with fs={fs}"
                    )
            else:
                if not (0 < Wn < 1):
                    raise ValueError(
                        "Normalized cutoff must satisfy 0 < fc < fs/2; "
                        f"got Wn={Wn} with fs={fs}"
                    )
            b, a = butter(order, Wn, btype=btype)
            y = filtfilt(b, a, y)
        # 3. Baseline correction
        if self.baseline_params:
            params = {**self.BASELINE_DEFAULTS, **self.baseline_params}
            y = detrend(y, **params)
        # 4. Trimming
        if self.trim_params:
            defaults = {"t_start": float(t[0]), "t_end": float(t[-1])}
            params = {**defaults, **self.trim_params}
            t_start = params["t_start"]
            t_end = params["t_end"]
            if t_end <= t_start:
                raise ValueError("t_end must be greater than t_start")
            mask = (t >= t_start) & (t <= t_end)
            if not np.any(mask):
                raise ValueError("Trim window does not overlap channel time range")
            t = t[mask]
            y = y[mask]
        # 5. Calibration
        if self.calibration_factor != 1.0:
            y = y * self.calibration_factor
        # Cache result
        if use_cache:
            self._cache_signature = signature
            self._cache_processed_time = t
            self._cache_processed_data = y
        return t, y

    def xy(
        self, processed: bool = True, use_cache: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to get (x, y) for plotting.

        Parameters
        ----------
        processed : bool, optional
            If True (default), returns the processed view via processed().
            If False, returns the raw time and raw data exactly as stored.
        use_cache : bool, optional
            Passed through to processed().
        """
        if processed:
            return self.processed(use_cache=use_cache)
        else:
            return self.time, self.data

    # ------------------------------------------------------------------ #
    # Simple time-domain peaks and RMS
    # ------------------------------------------------------------------ #

    def max_abs(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value where the data reaches its maximum absolute amplitude.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed view.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        t_peak : float
            Time at which |data| is maximum.
        y_peak : float
            Data value at that time (keeps original sign).
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute max_abs of empty signal")
        idx = int(np.argmax(np.abs(y)))
        return float(t[idx]), float(y[idx])

    def max_value(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value of the maximum data value (positive peak).

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed view.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        t_max : float
            Time at which data reaches its maximum value.
        y_max : float
            Maximum data value.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute max_value of empty signal")
        idx = int(np.argmax(y))
        return float(t[idx]), float(y[idx])

    def min_value(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value of the minimum data value (negative peak).

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed view.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        t_min : float
            Time at which data reaches its minimum value.
        y_min : float
            Minimum data value.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute min_value of empty signal")
        idx = int(np.argmin(y))
        return float(t[idx]), float(y[idx])

    def rms(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> float:
        """
        Compute the Root Mean Square (RMS) value of the channel.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed signal (including calibration).
            If False, use the raw data as stored.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        float
            RMS value of the selected signal.
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute RMS of empty signal")
        return float(np.sqrt(np.mean(y**2)))

    # ------------------------------------------------------------------ #
    # Fourier amplitude spectrum + peak
    # ------------------------------------------------------------------ #

    def fourier(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> FourierSpectrum:
        """
        Compute the (single-sided) Fourier amplitude spectrum of the channel.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed signal (via processed()).
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        FourierSpectrum
            Object containing frequency array `f` and amplitude spectrum `s`,
            with a `.peak()` helper.
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
        n = len(y)
        if n == 0:
            raise ValueError("Cannot compute Fourier spectrum of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Fourier transform requires a positive dt")
        n_fft = 1 << (n - 1).bit_length()
        f = np.fft.rfftfreq(n=n_fft, d=self.dt)
        s = np.abs(np.fft.rfft(y, n=n_fft))
        return FourierSpectrum(f=f, s=s)

    def fourier_peak(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Convenience wrapper returning the dominant frequency and its amplitude
        from the Fourier spectrum.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed signal.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        f_peak : float
            Frequency at the maximum amplitude in the spectrum.
        s_peak : float
            Maximum amplitude value.
        """
        spec = self.fourier(processed=processed, use_cache=use_cache)
        return spec.peak()

    # ------------------------------------------------------------------ #
    # Welch PSD + peak (MATLAB-ish defaults)
    # ------------------------------------------------------------------ #

    def welch_psd(
        self,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> WelchSpectrum:
        """
        Compute the Power Spectral Density (PSD) using Welch's method.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed signal.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().
        **kwargs
            Additional keyword arguments passed to scipy.signal.welch.
            If 'nperseg' is not given, a MATLAB-like default is used:
            nperseg = min(256, len(y)).

        Returns
        -------
        WelchSpectrum
            Object containing frequency array `f` and PSD values `p`,
            with a `.peak()` helper.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        n = len(y)
        if n == 0:
            raise ValueError("Cannot compute Welch PSD of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Welch PSD requires a positive dt")
        fs = 1.0 / self.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, p = welch(x=y, fs=fs, **kwargs)
        return WelchSpectrum(f=f, p=p)

    def welch_peak(
        self,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Convenience wrapper returning the dominant frequency and PSD amplitude
        from the Welch spectrum.

        Parameters
        ----------
        processed : bool, optional
            If True (default), use the processed signal.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().
        **kwargs
            Additional keyword arguments passed to scipy.signal.welch.

        Returns
        -------
        f_peak : float
            Frequency at which the PSD is maximum.
        p_peak : float
            Maximum PSD value.
        """
        psd = self.welch_psd(processed=processed, use_cache=use_cache, **kwargs)
        return psd.peak()

    # ------------------------------------------------------------------ #
    # Arias intensity
    # ------------------------------------------------------------------ #

    def arias_intensity(
        self,
        g: float = 9.81,
        processed: bool = True,
        use_cache: bool = True,
    ) -> AriasResult:
        """
        Compute the Arias intensity time history and significant duration.

        Assumes the channel data represents acceleration in units of g.
        The data is converted to m/s^2 internally using the factor `g`.

        Parameters
        ----------
        g : float, optional
            Acceleration due to gravity in m/s^2. Default is 9.81.
        processed : bool, optional
            If True (default), use the processed signal.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        AriasResult
            Structured result with time `t`, intensity `Ia`, final intensity,
            duration, and indices of 5% and 95% points.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute Arias intensity of empty signal")
        a_mps2 = g * y
        coef = np.pi / (2.0 * g)
        Ia = cumulative_trapezoid(coef * a_mps2**2, t, initial=0.0)
        Ia_final = float(Ia[-1])
        if Ia_final <= 0:
            return AriasResult(t=t, Ia=Ia, t_start=0, t_end=0)
        idx_start = int(np.argmax(Ia >= 0.05 * Ia_final))
        idx_end = int(np.argmax(Ia >= 0.95 * Ia_final))
        t_start = float(t[idx_start])
        t_end = float(t[idx_end])
        return AriasResult(
            t=t,
            Ia=Ia,
            t_start=t_start,
            t_end=t_end,
        )

    # ------------------------------------------------------------------ #
    # Response spectrum (SDOF, Newmark-beta average acceleration)
    # ------------------------------------------------------------------ #

    def _sdof_newmark_response(
        self,
        acc: np.ndarray,
        dt: float,
        omega: float,
        ksi: float,
    ) -> tuple[float, float, float]:
        """
        Internal helper: Newmark-beta (average acceleration) SDOF response to
        base acceleration.

        Parameters
        ----------
        acc : np.ndarray
            Ground acceleration time history a_g(t) (m/s^2).
        dt : float
            Time step (s).
        omega : float
            Circular frequency of the oscillator (rad/s).
        ksi : float
            Damping ratio.

        Returns
        -------
        Sd : float
            Peak relative displacement.
        Sv : float
            Peak relative velocity.
        Sa : float
            Peak absolute acceleration.
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

    def response_spectrum(
        self,
        periods: np.ndarray = np.linspace(0.05, 5.0, 100),
        ksi: float = 0.05,
        processed: bool = True,
        use_cache: bool = True,
    ) -> ResponseSpectrum:
        """
        Compute the elastic response spectrum for a family of SDOF oscillators
        subjected to this channel as base acceleration.

        Parameters
        ----------
        periods : np.ndarray
            Array of natural periods (s) for which to compute the spectrum.
        ksi : float, optional
            Damping ratio (e.g. 0.05 for 5% damping). Default is 0.05.
        processed : bool, optional
            If True (default), use the processed signal.
            If False, use the raw data.
        use_cache : bool, optional
            Passed through to processed().

        Returns
        -------
        ResponseSpectrum
            Structured response spectrum with Sd, Sv, Sa for each period.
        """
        t, a = self.xy(processed=processed, use_cache=use_cache)
        if a.size == 0:
            raise ValueError("Cannot compute response spectrum of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Response spectrum requires a positive dt")
        periods = np.asarray(periods, dtype=float)
        if periods.ndim != 1:
            raise ValueError("periods must be a 1D array")
        if np.any(periods <= 0.0):
            raise ValueError("All periods must be positive")
        Sd = np.zeros_like(periods, dtype=float)
        Sv = np.zeros_like(periods, dtype=float)
        Sa = np.zeros_like(periods, dtype=float)
        for i, T in enumerate(periods):
            omega = 2.0 * np.pi / T
            Sd[i], Sv[i], Sa[i] = self._sdof_newmark_response(a, self.dt, omega, ksi)
        return ResponseSpectrum(T=periods, Sd=Sd, Sv=Sv, Sa=Sa, ksi=ksi)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        processed: bool = True,
        use_cache: bool = True,
        include_label: bool = True,
        include_kind: bool = False,
        include_legend: bool = False,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the time history of this channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, a new figure and axes are created.
        processed : bool, optional
            If True (default), plot the processed view. If False, plot raw data.
        use_cache : bool, optional
            Passed through to processed().
        include_label : bool, optional
            If True (default), use the channel's label_axis for the y-axis label.
        include_kind : bool, optional
            If True, include the channel's quantity for the y-axis label.
            This overrides include_label if both are True.
        include_legend : bool, optional
            If True, use the channel's label_legend for the legend.
        **plot_kwargs
            Extra keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        t, y = self.xy(processed=processed, use_cache=use_cache)
        line_label = self.label_legend or self.name_user or self.name_input
        ax.plot(t, y, label=line_label, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        if include_label and self.label_axis:
            ylabel = self.label_axis
        if include_kind and self.quantity:
            ylabel = self.quantity.capitalize() + (
                f" [{self.units}]" if self.units else ""
            )
        ax.set_ylabel(ylabel)
        if include_legend and line_label:
            ax.legend()
        ax.grid(True)
        return ax
