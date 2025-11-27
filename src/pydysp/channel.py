from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.integrate import cumulative_trapezoid


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
            self.label_axis = f"{base} ({self.units})" if base else f"({self.units})"

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
        if self.dt:
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
        defaults = {"points": 50}
        params = {**defaults, **self.drift_params, **override}
        new = replace(self, drift_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"drift_params"})
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
        defaults = {"btype": "lowpass", "fc": 50.0, "order": 2}
        params = {**defaults, **self.filter_params, **override}
        new = replace(self, filter_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({f"filter_params"})
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
        defaults = {"type": "linear"}
        params = {**defaults, **self.baseline_params, **override}
        new = replace(self, baseline_params=params)
        new._clear_cache()
        arg_str = ", ".join(f"{k}={v}" for k, v in params.items()) or ""
        new.tags = set(self.tags).union({"baseline_params"})
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
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        new = replace(self, trim_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"trim_params"})
        new.processing_notes = [
            *self.processing_notes,
            f"Trim params set: {t_start}–{t_end} s",
        ]
        return new

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
            defaults = {"points": 50}
            params = {**defaults, **self.drift_params}
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
            defaults = {"btype": "lowpass", "fc": 50.0, "order": 2}
            params = {**defaults, **self.filter_params}
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
            b, a = butter(order, Wn, btype=btype)
            y = filtfilt(b, a, y)
        # 3. Baseline correction
        if self.baseline_params:
            defaults = {"type": "linear"}
            params = {**defaults, **self.baseline_params}
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

    def timehistory(self) -> tuple[np.ndarray, list[float]]:
        """
        Returns the time history data.

        Returns:
            np.ndarray: Array of time and scaled data values.
            list: Maximum time and data values.
        """
        t = self._time
        y = self._data * self.calibration
        index = np.argmax(np.abs(y))
        t_max = t[index]
        y_max = y[index]
        return np.array([t, y]), [t_max, y_max]

    def fourier(self) -> tuple[np.ndarray, list[float]]:
        """
        Computes the Fourier transform of the data.

        Returns:
            np.ndarray: Array of frequencies and Fourier amplitudes.
            list: Maximum frequency and amplitude values.
        """
        [t, y] = self.timehistory()[0]
        _no_freqs = int(2 ** (self._points - 1).bit_length())
        f = np.fft.rfftfreq(n=_no_freqs, d=self._timestep)
        s = np.abs(np.fft.rfft(a=y, n=_no_freqs))
        index = np.argmax(s)
        f_n = f[index]
        s_max = s[index]
        return np.array([f, s]), [f_n, s_max]

    def welch(self, **kwargs) -> tuple[np.ndarray, list[float]]:
        """
        Computes the Power Spectral Density using Welch's method.

        Parameters:
            **kwargs**: Additional keyword arguments to pass to scipy.signal.welch.

        Returns:
            np.ndarray: Array of frequencies and power spectral densities.
            list: Maximum frequency and power spectral density values.
        """
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = int(len(self._data) / 4.5)
        f, p = welch(x=self._data, fs=1 / self._timestep, **kwargs)
        index = np.argmax(p)
        f_n = f[index]
        p_max = p[index]
        return np.array([f, p]), [f_n, p_max]

    def arias(
        self, g: float = 9.81
    ) -> tuple[list[np.ndarray, np.ndarray], float, float, list[int]]:
        """
        Computes the Arias intensity.

        Parameters:
            g (float): Acceleration due to gravity.

        Returns:
            list: Time values and Arias intensity values.
            float: Final Arias intensity value.
            float: Duration of the significant shaking.
            list: Start and end indices for the significant shaking period.
        """
        arias = cumulative_trapezoid(
            x=self._time, y=np.pi / 2 / 9.81 * (g * self._data * self.calibration) ** 2
        )
        arias = np.append(arias, arias[-1])
        start = np.argmax(arias > 0.05 * arias[-1])
        end = np.argmax(arias > 0.95 * arias[-1])
        duration = self._time[end] - self._time[start]
        return [self._time, arias], arias[-1], duration, [start, end]

    def rms(self) -> float:
        """
        Computes the Root Mean Square (RMS) of the data.

        Returns:
            float: RMS value.
        """
        y = self._data * self.calibration
        return np.sqrt(np.mean(y**2))

    def plot(
        self,
        plot_type: str = "Timehistory",
        name: bool = True,
        description: bool = False,
        typey: bool = True,
        axis=None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plots the specified type of data.

        Parameters:
            plot_type (str): Type of plot ('Timehistory', 'Fourier', 'Power', 'Arias').
            name (bool): If True, includes the channel name in the ylabel.
            description (bool): If True, includes the channel description in the ylabel.
            typey (bool): If True, includes the plot type in the ylabel.
            axis: Matplotlib axis to plot on. If None, creates a new axis.
            **kwargs**: Additional keyword arguments for the plot.

        Returns:
            plt.Axes: The axis with the plotted data.
        """
        if axis is None:
            _, axis = plt.subplots()
        freq_plot = False
        match plot_type:
            case "Timehistory":
                [x, y] = self.timehistory()[0]
                xlabel = "Time (sec)"
                ytype = "Timehistory (" + self.unit + ")"
            case "Fourier":
                [x, y] = self.fourier()[0]
                xlabel = "Frequency (Hz)"
                ytype = "Fourier Amplitude"
                freq_plot = True
            case "Power":
                [x, y] = self.welch(**kwargs)[0]
                xlabel = "Frequency (Hz)"
                ytype = "Power Spectral Density"
                freq_plot = True
            case "Arias":
                [x, y] = self.arias()[0]
                xlabel = "Time (sec)"
                ytype = "Arias Intensity (m/s)"
            case _:
                raise ValueError(f"Unknown plot_type: {plot_type}")
        if freq_plot:
            axis.set_xlim(0, kwargs.get("xlim", 50))
        axis.plot(x, y)
        ylabel = ""
        if name:
            ylabel += self.name
        if description:
            ylabel += " " + self.description
        if typey:
            ylabel += " " + ytype
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid()
        return axis
