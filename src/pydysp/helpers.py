from __future__ import annotations

from typing import Any, Literal, Optional, Mapping

import matplotlib.pyplot as plt

from .channel import Channel
from .response import ResponseSpectrum


PlotType = Literal["time", "timehistory", "fourier", "psd", "welch"]


def annotate_peak(
    channel: Channel,
    ax: plt.Axes,
    plot: PlotType = "time",
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    welch_kwargs: Optional[Mapping[str, Any]] = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate a peak on a plot of a :class:`Channel`.

    This helper assumes the relevant quantity has already been plotted
    on ``ax`` using the corresponding ``Channel.plot*`` method.

    Parameters
    ----------
    channel :
        The channel whose peak will be annotated.
    ax :
        Matplotlib axes on which the data has already been plotted.
    plot : {"time", "timehistory", "fourier", "psd", "welch"}, optional
        Kind of plot that is currently shown on ``ax``:

        * "time" / "timehistory"  : time-history peak in the time domain.
        * "fourier"               : Fourier amplitude spectrum peak.
        * "psd" / "welch"         : Welch PSD peak.
    peak : {"abs", "max", "min"}, optional
        Which time-domain peak to use when ``plot`` is "time" or
        "timehistory":

        * "abs" : point where ``|y|`` is maximum (default).
        * "max" : maximum (most positive) value.
        * "min" : minimum (most negative) value.

        Ignored for Fourier / PSD plots.
    processed : bool, optional
        Use processed channel data (default True).
    use_cache : bool, optional
        Use the Channel processed-data cache (default True).
    fmt : str, optional
        Text format string for the annotation. If ``None`` (default),
        a sensible default is used depending on ``plot``:

        * time-domain: ``"{y:.3g} {units}"`` or ``"{y:.3g}"``.
        * Fourier   : ``"{f:.3g} Hz\\n{amp:.3g}"``.
        * PSD       : ``"{f:.3g} Hz\\n{p:.3g}"``.

        The format string is applied with ``str.format`` and can use:

        * time-domain: ``t``, ``y``.
        * Fourier   : ``f``, ``amp``.
        * PSD       : ``f``, ``p``.
    welch_kwargs : mapping, optional
        Extra keyword arguments forwarded to
        :meth:`Channel.welch_psd` when ``plot`` is "psd"/"welch".
    **annotate_kwargs :
        Additional keyword arguments forwarded to ``ax.annotate``,
        e.g. ``fontsize``, ``arrowprops``, ``ha``, ``va``, etc.
    """
    plot_norm = plot.lower()
    if plot_norm in {"time", "timehistory"}:
        _annotate_time_peak(
            channel=channel,
            ax=ax,
            peak=peak,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            **annotate_kwargs,
        )
    elif plot_norm == "fourier":
        _annotate_fourier_peak(
            channel=channel,
            ax=ax,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            **annotate_kwargs,
        )
    elif plot_norm in {"psd", "welch"}:
        _annotate_psd_peak(
            channel=channel,
            ax=ax,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            welch_kwargs=welch_kwargs,
            **annotate_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported plot type for annotate_peak: "
            f"{plot!r}. Expected 'time', 'timehistory', 'fourier', 'psd' or 'welch'."
        )


def _annotate_time_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    **annotate_kwargs: Any,
) -> None:
    if peak == "abs":
        t_peak, y_peak = channel.max_abs(processed=processed, use_cache=use_cache)
    elif peak == "max":
        t_peak, y_peak = channel.max_value(processed=processed, use_cache=use_cache)
    elif peak == "min":
        t_peak, y_peak = channel.min_value(processed=processed, use_cache=use_cache)
    else:
        raise ValueError("peak must be one of 'abs', 'max', or 'min'")

    if fmt is None:
        if channel.units:
            fmt = "{y:.3g} " + channel.units
        else:
            fmt = "{y:.3g}"

    text = fmt.format(t=t_peak, y=y_peak)

    ax.plot([t_peak], [y_peak], "o")
    ax.annotate(
        text,
        xy=(t_peak, y_peak),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        **annotate_kwargs,
    )


def _annotate_fourier_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    **annotate_kwargs: Any,
) -> None:
    spec = channel.fourier(processed=processed, use_cache=use_cache)
    f_peak, s_peak = spec.peak()

    if fmt is None:
        fmt = "{f:.3g} Hz\\n{amp:.3g}"

    text = fmt.format(f=f_peak, amp=s_peak)

    ax.plot([f_peak], [s_peak], "o")
    ax.annotate(
        text,
        xy=(f_peak, s_peak),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        **annotate_kwargs,
    )


def _annotate_psd_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    welch_kwargs: Optional[Mapping[str, Any]] = None,
    **annotate_kwargs: Any,
) -> None:
    psd = channel.welch_psd(
        processed=processed,
        use_cache=use_cache,
        **({} if welch_kwargs is None else dict(welch_kwargs)),
    )
    f_peak, p_peak = psd.peak()

    if fmt is None:
        fmt = "{f:.3g} Hz\\n{p:.3g}"

    text = fmt.format(f=f_peak, p=p_peak)

    ax.plot([f_peak], [p_peak], "o")
    ax.annotate(
        text,
        xy=(f_peak, p_peak),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        **annotate_kwargs,
    )


def annotate_time_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper around :func:`annotate_peak` for time histories.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="time",
        peak=peak,
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        **annotate_kwargs,
    )


def annotate_fourier_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper around :func:`annotate_peak` for Fourier spectra.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="fourier",
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        **annotate_kwargs,
    )


def annotate_psd_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: Optional[str] = None,
    welch_kwargs: Optional[Mapping[str, Any]] = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper around :func:`annotate_peak` for PSD (Welch) plots.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="psd",
        peak="abs",  # ignored by PSD branch
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        welch_kwargs=welch_kwargs,
        **annotate_kwargs,
    )


def annotate_response_peak(
    rs: ResponseSpectrum,
    ax: plt.Axes,
    *,
    fmt: str = "{T:.3g} s\\n{Sa:.3g} g",
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate the peak spectral acceleration on a response spectrum plot.

    This helper assumes a :class:`ResponseSpectrum` has already been
    plotted on ``ax`` via ``rs.plot(...)``.
    """
    T_peak, Sa_peak = rs.peak()

    text = fmt.format(T=T_peak, Sa=Sa_peak)

    ax.plot([T_peak], [Sa_peak], "o")
    ax.annotate(
        text,
        xy=(T_peak, Sa_peak),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        **annotate_kwargs,
    )
