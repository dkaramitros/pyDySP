from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

import csv
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tabulate import tabulate

from .channel import Channel

# Type aliases for channel selection / referencing
ChannelKey = Union[int, str, Channel]
ChannelSelector = Union[ChannelKey, Sequence[ChannelKey], slice, None]


@dataclass
class Test:
    """
    Represents a single experiment containing multiple time-history channels.

    Features
    --------
    - Store experiment-level metadata (name, description, source file, timestamp).
    - Manage a collection of `Channel` objects (selection, grouping, renaming). :contentReference[oaicite:0]{index=0}
    - Provide experiment-level processing:
        * Batch processing: drift correction, filtering, baseline correction, trimming.
        * Pairwise analysis: transfer functions, time delays, cross-spectra.
        * Basic modal identification (e.g. via FRFs / stabilization diagrams).
    - Provide multi-channel plotting utilities for quick visual inspection.
    - Provide convenient I/O constructors and exporters for common lab formats.
    """

    # Name of the experiment (e.g. 'Shaking table Test 07').
    name: str
    # Longer description.
    description: Optional[str] = None
    # Path or identifier of the primary raw data file from which this Test was built.
    source_file: Optional[str] = None
    # String representation of the test time/date
    timestamp: Optional[str] = None

    # Ordered list of Channel objects belonging to this experiment.
    channels: List[Channel] = field(default_factory=list)
    # Free-form tags for grouping and filtering tests.
    tags: set[str] = field(default_factory=set)
    # Free-form metadata dictionary.
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Normalise internal containers and enforce basic type assumptions.
        """
        # Always store channels as a list of Channel objects
        self.channels = list(self.channels)
        # Confirm all entries are Channel instances
        for i, ch in enumerate(self.channels):
            if not isinstance(ch, Channel):
                raise TypeError(f"Test.channels[{i}] is not a Channel instance.")
        # Normalise tags/meta to base container types
        self.tags = set(self.tags)
        self.meta = dict(self.meta)
        # Derive a sensible name from source_file if name is empty
        if not self.name and self.source_file:
            base = os.path.basename(self.source_file)
            self.name = os.path.splitext(base)[0]

    # ------------------------------------------------------------------ #
    # Basic collection-like behaviour
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """
        Number of channels in this Test.
        For the full (channels, timesteps) shape, use `test.shape`.
        """
        return len(self.channels)

    def __getitem__(self, key):
        """
        Channel lookup by index or name.

        - int  -> position in the channels list (0-based)
        - str  -> match against Channel.name_user or Channel.name_input
        """
        # Integer index
        if isinstance(key, int):
            return self.channels[key]
        # String key
        if isinstance(key, str):
            key_lower = key.lower()
            # First pass: exact case-insensitive match on name_user
            for ch in self.channels:
                if ch.name_user and ch.name_user.lower() == key_lower:
                    return ch
            # Second pass: exact case-insensitive match on name_input
            for ch in self.channels:
                if ch.name_input and ch.name_input.lower() == key_lower:
                    return ch
            raise KeyError(f"No channel found with name '{key}'")
        raise TypeError(
            f"Test.__getitem__ only supports int or str keys, got {type(key)!r}"
        )

    def iter_channels(
        self,
        selector: ChannelSelector = None,
        tags: Optional[Iterable[str]] = None,
        require_all_tags: bool = False,
    ) -> Iterable[Channel]:
        """
        Iterate over channels selected by index / name / slice / list,
        with optional filtering by channel tags.

        Parameters
        ----------
        selector :
            How to pick the initial set of channels:
            - None        -> all channels
            - int         -> single channel by index
            - str         -> single channel by name (via __getitem__)
            - Channel     -> that channel (if it belongs to this Test)
            - slice       -> slice of the channels list
            - Sequence[...] of the above
        tags :
            Optional iterable of tag strings. If provided, only channels whose
            `ch.tags` intersect (or contain) these will be yielded.
        require_all_tags :
            - False (default): channel is kept if it has *any* of the requested tags.
            - True: channel is kept only if it has *all* of the requested tags.
        """
        # Resolve the base set from `selector`
        if selector is None:
            base = list(self.channels)
        elif isinstance(selector, Channel):
            if selector in self.channels:
                base = [selector]
            else:
                raise ValueError("Channel is not part of this Test")
        elif isinstance(selector, (int, str)):
            base = [self[selector]]
        elif isinstance(selector, slice):
            base = self.channels[selector]
        elif isinstance(selector, Sequence):
            tmp: List[Channel] = []
            for key in selector:
                if isinstance(key, Channel):
                    if key not in self.channels:
                        raise ValueError("Channel in selector is not part of this Test")
                    tmp.append(key)
                else:
                    tmp.append(self[key])
            base = tmp
        else:
            raise TypeError(
                "Unsupported selector type for iter_channels: " f"{type(selector)!r}"
            )
        # Tag-based filtering
        if tags is None:
            for ch in base:
                yield ch
            return
        required = set(tags)
        for ch in base:
            ch_tags = getattr(ch, "tags", set())
            if not ch_tags:
                continue
            if require_all_tags:
                if required.issubset(ch_tags):
                    yield ch
            else:
                if ch_tags.intersection(required):
                    yield ch

    def channel_names(self) -> list[str]:
        """
        Return a list of preferred channel names for this Test.

        For each channel the priority is:
        - name_user
        - name_input
        - fallback: 'ch{index}' (e.g. 'ch0', 'ch1', ...)

        This is useful for quick inspection and for knowing what
        string keys you can use with test.channel["..."].
        """
        names: list[str] = []
        for i, ch in enumerate(self.channels):
            if getattr(ch, "name_user", None):
                names.append(ch.name_user)
            elif getattr(ch, "name_input", None):
                names.append(ch.name_input)
            else:
                names.append(f"ch{i}")
        return names

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def channel(self):
        """
        Convenience view over this Test's channels.

        Allows:
            test.channel[3]        -> 4th Channel (by index)
            test.channel["Acc1"]   -> Channel with matching name_input/name_user

        The underlying list is still available as `test.channels`.
        """
        return self

    @property
    def n_channels(self) -> int:
        """
        Return the number of channels.
        """
        return len(self.channels)

    @property
    def n_timesteps(self) -> int:
        """
        Return the number of timesteps (samples) per channel.

        Raises
        ------
        ValueError
            If channels have differing numbers of samples.
        """
        if not self.channels:
            return 0
        # Length of the first channel's data
        n0 = self.channels[0].data.shape[0]
        # Check all other channels match
        for ch in self.channels[1:]:
            n_i = ch.data.shape[0]
            if n_i != n0:
                raise ValueError("Channels have differing numbers of samples.")
        return n0

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the (n_channels, n_timesteps) shape of this Test.
        """
        return (self.n_channels, self.n_timesteps)

    @property
    def duration(self) -> float:
        """
        Total duration of the test, in seconds.

        Raises
        ------
        ValueError
            If channels have inconsistent durations.
        """
        if not self.channels:
            return 0.0
        # Determine duration of the first channel
        dur0 = self.channels[0].time[-1] - self.channels[0].time[0]
        # Check consistency with all other channels
        for ch in self.channels[1:]:
            dur_i = ch.time[-1] - ch.time[0]
            if not np.isclose(dur_i, dur0, rtol=1e-6, atol=1e-12):
                raise ValueError("Channels have inconsistent durations.")
        return float(dur0)

    @property
    def dt(self) -> float:
        """
        Sampling interval of the test, in seconds.

        Raises
        ------
        ValueError
            If channels have inconsistent dt values.
        """
        if not self.channels:
            raise ValueError("Cannot determine dt: this Test has no channels.")
        # Reference dt from the first channel
        dt0 = self.channels[0].dt
        # Check consistency with all other channels
        for ch in self.channels[1:]:
            if ch.dt != dt0:
                raise ValueError("Inconsistent dt across channels.")
        return dt0

    # ------------------------------------------------------------------ #
    # Info / reporting
    # ------------------------------------------------------------------ #

    def info(self) -> str:
        """
        Return a human-readable summary of this Test and its channels.

        The returned string is not auto-printed; use print(test.info())
        """
        lines: list[str] = []
        # Test-level metadata
        lines.append(f"Test: {self.name or '<unnamed>'}")
        if self.description:
            lines.append(f"  Description : {self.description}")
        if self.source_file:
            lines.append(f"  Source file : {self.source_file}")
        if self.timestamp:
            lines.append(f"  Timestamp   : {self.timestamp}")
        lines.append(f"  Channels    : {self.n_channels}")
        # Timesteps
        try:
            n_ts = self.n_timesteps
            lines.append(f"  Timesteps   : {n_ts}")
        except ValueError as err:
            lines.append(f"  Timesteps   : <inconsistent> ({err})")
        # Duration
        try:
            dur = self.duration
            lines.append(f"  Duration    : {dur:.6g} s")
        except ValueError as err:
            lines.append(f"  Duration    : <inconsistent> ({err})")
        # Sampling
        try:
            dt = self.dt
            fs = float("inf") if dt == 0 else 1.0 / dt
            lines.append(f"  Sampling    : dt={dt:.6g} s, fs={fs:.6g} Hz")
        except ValueError as err:
            lines.append(f"  Sampling    : <inconsistent> ({err})")
        # Tags and meta
        if self.tags:
            lines.append(f"  Test tags   : {', '.join(sorted(self.tags))}")
        if self.meta:
            meta_keys = ", ".join(sorted(self.meta.keys()))
            lines.append(f"  Meta keys   : {meta_keys}")
        # Channel table
        lines.append("")
        # Build table data: rows and headers
        headers = [
            "idx",
            "name_user",
            "name_input",
            "quantity",
            "units",
            "tags",
        ]
        table_rows = []
        for i, ch in enumerate(self.channels):
            table_rows.append(
                [
                    i,
                    ch.name_user or "-",
                    ch.name_input or "-",
                    ch.quantity or "-",
                    ch.units or "-",
                    ",".join(sorted(ch.tags)) if ch.tags else "-",
                ]
            )
        # Format table
        table_str = tabulate(
            table_rows,
            headers=headers,
            tablefmt="github",
            numalign="right",
            stralign="left",
        )
        lines.append(table_str)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Constructors (I/O)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_channels(
        cls,
        name: str,
        channels: Sequence[Channel],
        description: Optional[str] = None,
        source_file: Optional[str] = None,
        timestamp: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "Test":
        """
        Construct a Test directly from an existing sequence of Channel objects.

        Parameters
        ----------
        name :
            Human-readable name for this test (e.g. 'Test 07').
        channels :
            Sequence of Channel instances to include in the test.
        description :
            Optional description of the experiment.
        source_file :
            Optional path or identifier of the original data file.
        timestamp :
            Optional timestamp string for the experiment.
        tags :
            Optional iterable of test-level tags (e.g. {'sofsi', 'equals'}).
        meta :
            Optional additional metadata map (e.g. {'specimen_id': 'ABC123'}).

        Notes
        -----
        This is a thin convenience wrapper around the dataclass constructor:
        it normalises `channels`, `tags`, and `meta` into the expected container
        types and lets `__post_init__` do the remaining validation.
        """
        # Normalise containers
        channels_list = list(channels)
        tags_set = set(tags) if tags is not None else set()
        meta_dict = dict(meta) if meta is not None else {}
        # Delegate to the regular constructor
        return cls(
            name=name,
            description=description,
            source_file=source_file,
            timestamp=timestamp,
            channels=channels_list,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_sofsi_mat(
        cls,
        filename: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "Test":
        """
        Construct a Test from a SoFSI-style MATLAB .mat file.

        Expected format
        ---------------
        Required:
            Channel_1_Data       : 1D time vector (s)
            Channel_i_Data       : 1D data arrays, i >= 2

        Optional:
            File_Header          : struct with any subset of:
                                NumberOfChannels, NumberOfSamplesPerChannel,
                                SampleFrequency, Date, Comment,
                                NumberOfSamplesPerBlock, ...
            Channel_i_Header     : structs with fields SignalName, Unit,
                                MaxLevel, Correction, ...

        Header fields are used only for metadata.

        Parameters
        ----------
                Parameters
        ----------
        filename :
            Path to the MAT file.
        name :
            Optional test name. If not given, the stem of `filename` is used.
        description :
            Optional description. If not given, defaults to `name`.
        tags :
            Optional iterable of test-level tags (e.g. {"demo"}).
        meta :
            Optional mapping to initialise Test.meta with (e.g. {"specimen": "ABC"}).
        """
        try:
            imported_data = sp.io.loadmat(
                filename, squeeze_me=True, struct_as_record=False
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File '{filename}' not found.") from exc

        # Helpers for optional header parsing
        def _parse_int(val):
            try:
                return int(val)
            except Exception:
                return None

        def _parse_float(val):
            try:
                return float(val)
            except Exception:
                return None

        def _get_attr(obj, field: str):
            if obj is None:
                return None
            return getattr(obj, field, None)

        # 1. File_Header
        header_dict: dict[str, Any] = {}
        file_header = imported_data.get("File_Header", None)
        if file_header is not None:
            header_dict = {
                "NumberOfChannels": _parse_int(
                    _get_attr(file_header, "NumberOfChannels")
                ),
                "NumberOfSamplesPerChannel": _parse_int(
                    _get_attr(file_header, "NumberOfSamplesPerChannel")
                ),
                "SampleFrequency": _parse_float(
                    _get_attr(file_header, "SampleFrequency")
                ),
                "Date": _get_attr(file_header, "Date"),
                "Comment": _get_attr(file_header, "Comment"),
                "NumberOfSamplesPerBlock": _get_attr(
                    file_header, "NumberOfSamplesPerBlock"
                ),
            }
        # 2. Time vector: Channel_1_Data
        if "Channel_1_Data" not in imported_data:
            raise KeyError(
                "SoFSI MAT file must contain 'Channel_1_Data' as time channel."
            )
        time_vec = np.asarray(imported_data["Channel_1_Data"]).flatten()
        if time_vec.ndim != 1:
            raise ValueError(
                f"'Channel_1_Data' must be 1D (time vector), got shape {time_vec.shape!r}."
            )
        n_samples = len(time_vec)
        # 3. Data channels: Channel_i_Data (i >= 2)
        data_channels_indices: list[int] = []
        for key in imported_data.keys():
            if key.startswith("Channel_") and key.endswith("_Data"):
                try:
                    idx = int(key.split("_")[1])
                except Exception:
                    continue
                if idx >= 2:
                    data_channels_indices.append(idx)
        if not data_channels_indices:
            raise ValueError(
                "No data channels found in SoFSI MAT file "
                "(no 'Channel_i_Data' with i >= 2)."
            )
        data_channels_indices = sorted(set(data_channels_indices))
        channels: list[Channel] = []
        for idx in data_channels_indices:
            header_key = f"Channel_{idx}_Header"
            data_key = f"Channel_{idx}_Data"
            # Channel header
            hdr = imported_data.get(header_key, None)
            signal_name = _get_attr(hdr, "SignalName") or f"CH{idx-1}"
            unit = _get_attr(hdr, "Unit")
            max_level = _get_attr(hdr, "MaxLevel")
            correction = _get_attr(hdr, "Correction")
            ch_meta: dict[str, Any] = {}
            if max_level is not None:
                parsed_ml = _parse_float(max_level)
                ch_meta["max_level"] = parsed_ml if parsed_ml is not None else max_level
            if correction is not None:
                ch_meta["correction"] = correction
            # Channel data
            data_arr = np.asarray(imported_data[data_key]).flatten()
            if data_arr.ndim != 1:
                raise ValueError(
                    f"'{data_key}' must be 1D (n_samples,), got shape {data_arr.shape!r}."
                )
            if len(data_arr) != n_samples:
                raise ValueError(
                    f"Length of '{data_key}' ({len(data_arr)}) does not match "
                    f"length of Channel_1_Data ({n_samples})."
                )
            # Build Channel
            ch = Channel(
                data=data_arr,
                time=time_vec,
                name_input=str(signal_name),
                units=str(unit).strip() if unit is not None else None,
                raw_units=str(unit).strip() if unit is not None else None,
                meta=ch_meta,
            )
            channels.append(ch)
        if not channels:
            raise ValueError(
                "No data channels found in SoFSI MAT file "
                "(no 'Channel_i_Data' with i >= 2)."
            )
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        timestamp = header_dict.get("Date", None)
        tags_set = set(tags) if tags is not None else set()
        tags_set.add("sofsi")
        meta_dict = dict(meta) if meta is not None else {}
        meta_dict.setdefault("sofsi_file_header", header_dict)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=timestamp,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_equals_mat(
        cls,
        filename: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "Test":
        """
        Construct a Test from an EQUALS-style MATLAB .mat file.

        Expected format
        ---------------
        Required:
            t       : 1D numeric time vector (s)
            output  : 2D numeric array, shape (n_samples, n_channels)

        Optional:
            Testdate, Time, Frequency, Points, No_Channels, File_name,
            Buffer_Size, sampling, Filter, P_ref, ...

        Header fields are used only for metadata.

        Parameters
        ----------
        filename :
            Path to the MAT file.
        name :
            Optional test name. If not given, the stem of `filename` is used.
        description :
            Optional description. If not given, defaults to `name`.
        tags :
            Optional iterable of test-level tags (e.g. {"demo"}).
        meta :
            Optional mapping to initialise Test.meta with (e.g. {"specimen": "ABC"}).
        """
        try:
            imported_data = sp.io.loadmat(
                filename, squeeze_me=True, struct_as_record=False
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File '{filename}' not found.") from exc

        # Helpers for optional header parsing
        def _get(key: str):
            return imported_data.get(key, None)

        def _parse_int(val):
            try:
                return int(val)
            except Exception:
                return None

        def _parse_float(val):
            try:
                return float(val)
            except Exception:
                return None

        # 1. Header metadata
        header_dict: dict[str, Any] = {}
        header_dict = {
            "Testdate": _get("Testdate"),
            "Time": _get("Time"),
            "Frequency": _parse_float(_get("Frequency")),
            "Points": _parse_int(_get("Points")),
            "No_Channels": _parse_int(_get("No_Channels")),
            "File_name": _get("File_name"),
            "Buffer_Size": _parse_int(_get("Buffer_Size")),
            "sampling": _get("sampling"),
            "Filter": _get("Filter"),
            "P_ref": _parse_float(_get("P_ref")),
        }
        # 2. Time vector: t
        if "t" not in imported_data:
            raise KeyError("EQUALS MAT file must contain time vector 't'.")
        time_vec = np.asarray(imported_data["t"]).flatten()
        if time_vec.ndim != 1:
            raise ValueError(f"'t' must be 1D, got shape {time_vec.shape!r}.")
        # 3. Data matrix: output
        if "output" not in imported_data:
            raise KeyError("EQUALS MAT file must contain data matrix 'output'.")
        data_matrix = np.asarray(imported_data["output"])
        if data_matrix.ndim != 2:
            raise ValueError(
                f"'output' must be 2D (n_samples × n_channels), "
                f"got shape {data_matrix.shape!r}."
            )
        n_samples, n_channels = data_matrix.shape
        if len(time_vec) != n_samples:
            raise ValueError(
                f"Time vector length ({len(time_vec)}) does not match "
                f"rows of 'output' ({n_samples})."
            )
        # Build channels
        channels: list[Channel] = []
        for i in range(n_channels):
            ch = Channel(
                data=np.asarray(data_matrix[:, i]).flatten(),
                time=time_vec,
                name_input=f"CH{i+1}",
                name_user=f"CH{i+1}",
            )
            channels.append(ch)
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        testdate = header_dict["Testdate"]
        timestr = header_dict["Time"]
        timestamp: Optional[str] = None
        if isinstance(testdate, str) and isinstance(timestr, str):
            timestamp = f"{testdate} {timestr}"
        elif isinstance(testdate, str):
            timestamp = testdate
        tags_set = set(tags) if tags is not None else set()
        tags_set.add("equals")
        meta_dict = dict(meta) if meta is not None else {}
        meta_dict.setdefault("equals_header", header_dict)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=timestamp,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_csv(
        cls,
        filename: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "Test":
        """
        Construct a Test from a CSV file.

        Expected format
        ---------------
        - First row is a header.
        - First column is time ('Time').
        - Remaining columns are channels.
        - All values are numeric.

        Parameters
        ----------
        filename :
            Path to the CSV file.
        name :
            Optional test name. If not given, the stem of `filename` is used.
        description :
            Optional description. If not given, defaults to `name`.
        tags :
            Optional iterable of test-level tags (e.g. {"csv", "demo"}).
        meta :
            Optional mapping to initialise Test.meta with (e.g. {"specimen": "ABC"}).
        """
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # must exist
            rows = [row for row in reader]
        if not header:
            raise ValueError("CSV file has no header row.")
        if len(header) < 2:
            raise ValueError("CSV must have Time + at least one data column.")
        if header[0].strip().lower() != "time":
            raise ValueError("First column must be named 'Time'.")
        # Data matrix
        data = np.asarray(rows, dtype=float)
        time_vec = data[:, 0]
        data_cols = data[:, 1:]
        n_samples, n_channels = data_cols.shape
        # Build Channels
        channels: list[Channel] = []
        for i in range(n_channels):
            col_name = header[i + 1].strip() or f"CH{i+1}"
            ch = Channel(
                data=data_cols[:, i],
                time=time_vec,
                name_input=col_name,
                name_user=col_name,
            )
            channels.append(ch)
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        tags_set = set(tags) if tags else set()
        tags_set.add("csv")
        meta_dict = dict(meta) if meta else {}
        meta_dict.setdefault("csv_header", header)
        meta_dict.setdefault("csv_n_samples", n_samples)
        meta_dict.setdefault("csv_n_channels", n_channels)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=None,
            tags=tags_set,
            meta=meta_dict,
        )

    def to_csv(
        self,
        filename: str,
        selector: ChannelSelector = None,
        include_axis_labels: bool = True,
        overwrite: bool = True,
    ) -> None:
        """
        Export selected channels to a CSV file:
            Time, Ch1, Ch2, ...

        Parameters
        ----------
        filename : str
            Output CSV file path.
        selector :
            Which channels to export (index, name, list, slice…). If None,
            all channels are exported.
        include_axis_labels : bool
            If True, use channel.name_user (preferred) or channel.name_input
            as column names. Otherwise fallback to Ch1, Ch2, ...
        overwrite : bool
            If False and file exists, raise an error.
        """
        # Check overwrite
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        # Resolve channels
        selected_channels = list(self.iter_channels(selector))
        if not selected_channels:
            raise ValueError("No channels selected for CSV export.")
        # Time vector
        t0 = selected_channels[0].time
        for ch in selected_channels[1:]:
            if not np.array_equal(ch.time, t0):
                raise ValueError(
                    "All selected channels must share the same time vector "
                    "to export as a single CSV table."
                )
        # Build headers
        headers: list[str] = ["Time"]
        for i, ch in enumerate(selected_channels):
            if include_axis_labels:
                label = ch.label_axis or ch.name_user or ch.name_input or f"Ch{i+1}"
                headers.append(label)
            else:
                headers.append(f"Ch{i+1}")
        # Build rows
        data_matrix = np.column_stack([t0] + [ch.data for ch in selected_channels])
        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data_matrix)

    # ------------------------------------------------------------------ #
    # Channel management
    # ------------------------------------------------------------------ #

    def add_channel(self, ch: Channel) -> "Test":
        """
        Return a new Test with `ch` appended to the channels list.

        Functional style: Test is treated as immutable; this method returns
        a new instance rather than mutating in place.
        """
        # TODO: use `replace` from dataclasses to keep things immutable-ish
        raise NotImplementedError

    def with_channels(
        self,
        channels: Sequence[Channel],
        *,
        extend: bool = False,
    ) -> "Test":
        """
        Return a new Test with a modified channels list.

        - If extend=False: replace existing channels with the provided list.
        - If extend=True: append provided channels to the existing ones.
        """
        # TODO: construct new Test with updated channels
        raise NotImplementedError

    def drop_channels(self, selector: ChannelSelector) -> "Test":
        """
        Return a new Test with the selected channels removed.
        """
        # TODO: resolve selector and filter out matching channels
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Quality checks and consistency
    # ------------------------------------------------------------------ #

    def check_sampling_consistency(
        self,
        *,
        rtol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Inspect all channels and check that:

        - All dt are defined and consistent within relative tolerance rtol.
        - Time vectors are aligned (same length and same start/end times).

        Returns a small report dictionary with flags and diagnostics.
        """
        # TODO: compute and return consistency report
        raise NotImplementedError

    def detect_gaps_or_nans(
        self,
        selector: ChannelSelector = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Scan selected channels for NaNs, infs, or suspicious gaps in time.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Mapping from channel index to a small report dict
            (e.g. counts of NaNs, indices, gap stats).
        """
        # TODO: implement simple checks for data quality per channel
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Batch processing (experiment-level wrappers)
    # ------------------------------------------------------------------ #

    def drift_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        their drift-corrected versions (using Channel.drift_corrected). :contentReference[oaicite:4]{index=4}
        """
        # TODO: map selector → list of indices and apply Channel.drift_corrected
        raise NotImplementedError

    def filtered(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        their filtered versions (using Channel.filtered).
        """
        # TODO: call Channel.filtered on each selected channel
        raise NotImplementedError

    def baseline_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        baseline-corrected versions (using Channel.baseline_corrected).
        """
        # TODO: call Channel.baseline_corrected on each selected channel
        raise NotImplementedError

    def trimmed(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        trimmed versions (using Channel.trimmed).

        This is the generic "manual window" trimming interface.
        """
        # TODO: call Channel.trimmed on each selected channel
        raise NotImplementedError

    def trimmed_by_threshold(
        self,
        selector: ChannelSelector = None,
        *,
        threshold: float = 0.01,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed using the
        classic bracketed-duration threshold method. :contentReference[oaicite:5]{index=5}

        Delegates to Channel.trim_by_threshold for each channel, but you may
        later choose to align windows based on a reference channel.
        """
        # TODO: implement strategy (e.g. derive window from ref channel, apply to all)
        raise NotImplementedError

    def trimmed_by_fraction_of_peak(
        self,
        selector: ChannelSelector = None,
        *,
        fraction: float = 0.05,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed to the time window
        where the signal exceeds a fraction of its peak amplitude.
        """
        # TODO: delegate to Channel.trim_by_fraction_of_peak with a consistent strategy
        raise NotImplementedError

    def trimmed_by_arias(
        self,
        selector: ChannelSelector = None,
        *,
        lower: float = 0.05,
        upper: float = 0.95,
        g: float = 9.81,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed to the Arias-intensity
        significant duration window (e.g. 5–95% of Arias intensity). :contentReference[oaicite:6]{index=6}
        """
        # TODO: use Channel.trim_by_arias, potentially aligning on a reference channel
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Pairwise analysis (transfer functions, time delay, cross-spectra)
    # ------------------------------------------------------------------ #

    def cross_spectrum(
        self,
        from_ch: ChannelKey,
        to_ch: ChannelKey,
        *,
        processed: bool = True,
        use_cache: bool = True,
        **welch_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-spectrum (CSD) between two channels using Welch-like estimates.

        Returns a (f, Pxy) tuple.
        """
        # TODO: fetch x(t), y(t) from the two channels and call scipy.signal.csd
        raise NotImplementedError

    def transfer_function(
        self,
        from_ch: ChannelKey,
        to_ch: ChannelKey,
        *,
        method: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        **welch_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a single-input single-output transfer function between two channels.

        Returns
        -------
        f : np.ndarray
            Frequency axis (Hz).
        H : np.ndarray
            Complex transfer function values.
        coh : np.ndarray
            Magnitude-squared coherence.
        """
        # TODO: use PSDs + CSD to compute transfer function and coherence
        raise NotImplementedError

    def time_delay(
        self,
        from_ch: ChannelKey,
        to_ch: ChannelKey,
        *,
        method: Literal["xcorr", "argmax_phase"] = "xcorr",
        processed: bool = True,
        use_cache: bool = True,
        max_lag: Optional[float] = None,
    ) -> float:
        """
        Estimate time delay between two channels.

        Methods:
        - 'xcorr': use time-domain cross-correlation and take lag of max peak.
        - 'argmax_phase': infer delay from linear phase of transfer function.
        """
        # TODO: implement one or both strategies for delay estimation
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Basic modal identification (skeleton only)
    # ------------------------------------------------------------------ #

    def modal_identification(
        self,
        selector: ChannelSelector = None,
        *,
        method: Literal["ssi-cov", "ssi-data", "peak-picking"] = "ssi-cov",
        order_range: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """
        Perform basic output-only or input-output modal identification.

        The exact implementation (e.g. stochastic subspace identification,
        frequency-domain peak picking) will be added later.

        Returns a generic dictionary with e.g. natural frequencies, damping
        ratios, and (optionally) mode shapes and a stabilization diagram object.
        """
        # TODO: design and implement modal identification workflow
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Multi-channel plotting
    # ------------------------------------------------------------------ #

    def plot_timehistories(
        self,
        selector: ChannelSelector = None,
        *,
        columns: int = 1,
        sharex: bool = True,
        sharey: bool = False,
        processed: bool = True,
        use_cache: bool = True,
        include_labels: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        **plot_kwargs: Any,
    ) -> np.ndarray:
        """
        Plot time histories of selected channels in a grid of subplots.

        - Uses Channel.plot under the hood.
        - Arranges channels in `rows x columns` layout.
        - Returns the array of axes.
        """
        # TODO: create figure/subplots grid, loop over channels and call Channel.plot
        raise NotImplementedError

    def plot_spectra(
        self,
        selector: ChannelSelector = None,
        *,
        kind: Literal["fourier", "psd"] = "fourier",
        fmax: Optional[float] = 50.0,
        processed: bool = True,
        use_cache: bool = True,
        columns: int = 1,
        sharex: bool = True,
        sharey: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        **plot_kwargs: Any,
    ) -> np.ndarray:
        """
        Plot Fourier amplitude spectra or Welch PSDs for selected channels.

        Delegates to Channel.plot_fourier or Channel.plot_psd.
        """
        # TODO: similar layout logic to plot_timehistories, but call spectral plotting methods
        raise NotImplementedError

    def plot_transfer_function(
        self,
        from_ch: ChannelKey,
        to_ch: ChannelKey,
        *,
        method: Literal["H1", "H2"] = "H1",
        fmax: Optional[float] = None,
        processed: bool = True,
        use_cache: bool = True,
        axes: Optional[Sequence[plt.Axes]] = None,
        **welch_kwargs: Any,
    ) -> Sequence[plt.Axes]:
        """
        Convenience method to plot transfer function magnitude, phase, and coherence
        between two channels on one or more axes.
        """
        # TODO: compute transfer function via self.transfer_function and then plot on axes
        raise NotImplementedError
