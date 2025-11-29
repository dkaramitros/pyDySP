# channel.py
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.integrate import cumulative_trapezoid

from .spectra import FourierSpectrum, WelchSpectrum
from .arias import AriasResult
from .response import ResponseSpectrum, sdof_newmark_response


@dataclass(eq=False)
class Channel:
    """Single time-history channel with metadata and processing parameters.

    Parameters
    ----------
    data : np.ndarray
        1-D array of raw signal values.
    dt : float, optional
        Sampling interval in seconds. If not provided, a ``time`` array
        must be supplied.
    t0 : float, optional
        Start time in seconds (used when constructing ``time`` from ``dt``).
    time : np.ndarray, optional
        Explicit time array matching ``data`` in shape.

    Notes
    -----
    The class stores processing parameters (drift, filter, baseline, trim)
    and supports non-destructive processing: methods that change processing
    parameters return new ``Channel`` instances. Use ``processed()`` to
    obtain the time and data arrays with all current processing applied.
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
            raise ValueError(
                f"Channel.data must be a 1D array (single time history), got shape {self.data.shape!r}"
            )

        # Handle time / dt relationship
        if self.time is not None:
            self.time = np.asarray(self.time)
            if self.time.shape != self.data.shape:
                raise ValueError(
                    f"Time and data must have the same shape; got time.shape={self.time.shape!r}, data.shape={self.data.shape!r}"
                )
            if self.dt is None and len(self.time) > 1:
                self.dt = float(self.time[1] - self.time[0])
            if self.t0 == 0.0 and len(self.time) > 0:
                self.t0 = float(self.time[0])
        else:
            if self.dt is None:
                raise ValueError("Either 'time' or 'dt' must be provided")
            if self.dt <= 0:
                raise ValueError(f"dt must be positive, got {self.dt!r}")
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
            units = f"[{self.units}]"
            self.label_axis = f"{base} {units}".strip()

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        """
        Total duration of the channel in seconds, based on the time vector.
        """
        if self.data.size == 0:
            return 0.0
        t = self.time
        return float(t[-1] - t[0])

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
        """Return a clean, human-readable summary of channel metadata.

        Returns
        -------
        str
            Multi-line human-readable summary suitable for printing.
        """
        lines = []
        # Header
        title = self.name_user or self.name_input or "<unnamed>"
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
        """Return a new Channel with updated drift correction parameters.

        Parameters
        ----------
        **override :
            Keyword arguments forwarded to the stored drift parameters.

        Notes
        -----
        The actual correction is applied lazily in ``processed()``; this
        method only updates the parameters and returns a new ``Channel``.
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
        """Return a new Channel with updated Butterworth filter parameters.

        Parameters
        ----------
        **override :
            Keyword arguments forwarded to the stored filter parameters.

        Notes
        -----
        Actual filtering is applied in ``processed()``; this method only
        records the requested filter specification.
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
        """Return a new Channel with updated baseline correction parameters.

        Parameters
        ----------
        **override :
            Keyword arguments forwarded to the baseline correction method
            (passed to ``scipy.signal.detrend`` when applied).
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
        """Return a new Channel with updated trim window parameters.

        Parameters
        ----------
        **override :
            Keyword arguments typically including ``t_start`` and ``t_end``
            (in seconds) that define the trimming window.
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
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal: channel has zero samples")
        if use_abs:
            mask = np.abs(y) >= threshold
        else:
            mask = y >= threshold
        if not np.any(mask):
            raise ValueError(f"No samples exceed the specified threshold ({threshold})")
        i_start = int(np.argmax(mask))
        i_end = int(len(mask) - 1 - np.argmax(mask[::-1]))
        t_start = float(t[i_start]) - buffer_before
        t_end = float(t[i_end]) + buffer_after
        # Clamp to original time range
        t_start = max(t_start, float(t[0]))
        t_end = min(t_end, float(t[-1]))
        if t_end <= t_start:
            raise ValueError(
                f"Computed trim window is empty after buffering: t_start={t_start}, t_end={t_end}"
            )
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
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        _, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal: channel has zero samples")
        if use_abs:
            peak = float(np.max(np.abs(y)))
        else:
            peak = float(np.max(y))
        if peak <= 0.0:
            raise ValueError(
                f"Signal peak is non-positive ({peak}); cannot define threshold from peak"
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

        Data is assumed to be acceleration in g.
        """
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(
                f"Require 0 <= lower < upper <= 1, got lower={lower!r}, upper={upper!r}"
            )
        res = self.arias_intensity(g=g, processed=processed, use_cache=use_cache)
        t = res.t
        Ia = res.Ia
        if Ia.size == 0:
            raise ValueError("Cannot trim by Arias intensity for empty signal")
        Ia_final = float(Ia[-1])
        if Ia_final <= 0.0:
            raise ValueError(f"Final Arias intensity is non-positive ({Ia_final})")
        # Find indices where cumulative Arias crosses the requested fractions
        idx_lower = int(np.argmax(Ia >= lower * Ia_final))
        idx_upper = int(np.argmax(Ia >= upper * Ia_final))
        t_lower = float(t[idx_lower]) - buffer_before
        t_upper = float(t[idx_upper]) + buffer_after
        # Clamp to original time range
        t_lower = max(t_lower, float(t[0]))
        t_upper = min(t_upper, float(t[-1]))
        if t_upper <= t_lower:
            raise ValueError(
                f"Computed Arias trim window is empty after buffering: t_start={t_lower}, t_end={t_upper}"
            )
        return self.trimmed(t_start=t_lower, t_end=t_upper)

    # ------------------------------------------------------------------ #
    # Processed
    # ------------------------------------------------------------------ #

    def processed(self, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(time, data)`` after applying current processing steps.

        Processing order
        ----------------
        1. Drift correction
        2. Butterworth filter
        3. Baseline detrend
        4. Trimming (``t_start``, ``t_end``)
        5. Calibration-factor scaling

        Parameters
        ----------
        use_cache : bool, optional
            If True (default), cached processed results are returned when
            available and the processing signature matches.

        Returns
        -------
        t, y : tuple
            Time and data arrays after processing.
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
                    f"Number of points for drift correction ({points}) exceeds data length ({len(y)})"
                )
            drift_value = float(np.mean(y[:points]))
            y = y - drift_value
        # 2. Filtering
        if self.filter_params:
            if self.dt is None or self.dt <= 0:
                raise ValueError(
                    f"Filtering requires a positive dt, got dt={self.dt!r}"
                )
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
                    raise ValueError(
                        f"Band filters require f1 and f2, got f1={f1!r}, f2={f2!r}"
                    )
                Wn = [2 * f1 / fs, 2 * f2 / fs]
            else:
                raise ValueError(f"Unsupported filter mode: {btype!r}")
            if isinstance(Wn, (list, tuple)):
                if not (0 < Wn[0] < Wn[1] < 1):
                    raise ValueError(
                        f"Normalized band edges must satisfy 0 < f1 < f2 < fs/2; got Wn={Wn} with fs={fs}"
                    )
            else:
                if not (0 < Wn < 1):
                    raise ValueError(
                        f"Normalized cutoff must satisfy 0 < fc < fs/2; got Wn={Wn} with fs={fs}"
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
                raise ValueError(
                    f"t_end must be greater than t_start: t_start={t_start}, t_end={t_end}"
                )
            mask = (t >= t_start) & (t <= t_end)
            if not np.any(mask):
                raise ValueError(
                    f"Trim window [{t_start}, {t_end}] does not overlap channel time range [{float(t[0])}, {float(t[-1])}]"
                )
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
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
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
        Compute cumulative Arias intensity and significant-duration window.

        Assumes channel contains acceleration in g. Returns AriasResult with
        Ia(t) and t_start/t_end for the 5%/95% points by default.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute Arias intensity of empty signal")
        a_mps2 = g * y
        coef = np.pi / (2.0 * g)
        Ia = cumulative_trapezoid(coef * a_mps2**2, t, initial=0.0)
        Ia_final = float(Ia[-1])
        if Ia_final <= 0:
            raise ValueError("Final Arias intensity is non-positive")
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

    def response_spectrum(
        self,
        periods: np.ndarray = np.linspace(0.05, 5.0, 100),
        g: float = 9.81,
        ksi: float = 0.05,
        processed: bool = True,
        use_cache: bool = True,
    ) -> ResponseSpectrum:
        """
        Compute elastic response spectrum for a grid of periods.

        Inputs:
        - periods: 1D array of natural periods [s]
        - g: gravity to convert from g -> m/s^2
        - ksi: damping ratio

        Returns a ResponseSpectrum (Sa returned in g).
        """
        _, a = self.xy(processed=processed, use_cache=use_cache)
        if a.size == 0:
            raise ValueError("Cannot compute response spectrum of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Response spectrum requires a positive dt")
        a_mps2 = g * a  # convert g to m/s^2
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
            Sd[i], Sv[i], Sa[i] = sdof_newmark_response(a_mps2, self.dt, omega, ksi)
        Sa_g = Sa / g  # convert back to g
        return ResponseSpectrum(T=periods, Sd=Sd, Sv=Sv, Sa=Sa_g, ksi=ksi)

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
        """
        if ax is None:
            _, ax = plt.subplots()
        t, y = self.xy(processed=processed, use_cache=use_cache)
        line_label = self.label_legend or self.name_user or self.name_input
        ax.plot(t, y, label=line_label, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ylabel = ax.get_ylabel() or ""
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

    def plot_fourier(
        self,
        ax: Optional[plt.Axes] = None,
        processed: bool = True,
        use_cache: bool = True,
        fmax: Optional[float] = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Fourier amplitude spectrum of this channel.
        """
        spec = self.fourier(processed=processed, use_cache=use_cache)
        return spec.plot(ax=ax, fmax=fmax, **plot_kwargs)

    def plot_psd(
        self,
        ax: Optional[plt.Axes] = None,
        processed: bool = True,
        use_cache: bool = True,
        fmax: Optional[float] = 50.0,
        **welch_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Welch power spectral density of this channel.
        """
        spec = self.welch_psd(processed=processed, use_cache=use_cache, **welch_kwargs)
        return spec.plot(ax=ax, fmax=fmax)

    def plot_arias(
        self,
        ax: Optional[plt.Axes] = None,
        g: float = 9.81,
        processed: bool = True,
        use_cache: bool = True,
        show_window: bool = True,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Arias intensity time history (Husid plot) for this channel.

        Data is assumed to be acceleration in g.
        """
        res = self.arias_intensity(g=g, processed=processed, use_cache=use_cache)
        return res.plot(ax=ax, show_window=show_window, **plot_kwargs)

    def plot_response_spectrum(
        self,
        ax: Optional[plt.Axes] = None,
        periods: np.ndarray = np.linspace(0.05, 5.0, 100),
        ksi: float = 0.05,
        processed: bool = True,
        use_cache: bool = True,
        y: str = "Sa",
        logx: bool = False,
        logy: bool = False,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the elastic response spectrum for this channel.
        """
        rs = self.response_spectrum(
            periods=periods,
            ksi=ksi,
            processed=processed,
            use_cache=use_cache,
        )
        return rs.plot(
            ax=ax,
            y=y,
            logx=logx,
            logy=logy,
            **plot_kwargs,
        )
