from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
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
    Experiment container for multiple time-history channels.

    Holds metadata and an ordered list of Channel objects and provides:
    - selection helpers and tags,
    - batch processing (drift, filter, baseline, trim),
    - pairwise analyses (cross-spectrum, transfer function, time delay),
    - basic EMA model construction, plotting and common I/O (MAT/CSV).
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

    def __getitem__(self, key) -> Channel:
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
    def channel(self) -> "Test":
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
            if not np.isclose(ch.dt, dt0, rtol=1e-6, atol=1e-12):
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
        if not isinstance(ch, Channel):
            raise TypeError("add_channel expects a Channel instance.")
        new_channels = list(self.channels)
        new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def drop_channels(self, selector: ChannelSelector) -> "Test":
        """
        Return a new Test with the selected channels removed.

        The selector can be anything accepted by `iter_channels`, e.g.:
        - int           -> index
        - str           -> name (name_user / name_input)
        - Channel       -> that channel
        - slice         -> slice of the channels list
        - Sequence[...] -> list of the above
        """
        if selector is None:
            raise ValueError("drop_channels requires a non-None selector.")
        # Resolve which Channel objects to remove
        channels_to_drop = list(self.iter_channels(selector))
        # Keep all channels that are not in channels_to_drop
        new_channels = [ch for ch in self.channels if ch not in channels_to_drop]
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Batch processing
    # ------------------------------------------------------------------ #

    def drift_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        their drift-corrected versions (using Channel.drift_corrected).

        Parameters
        ----------
        selector :
            Which channels to process (index, name, Channel, list, slice…).
            If None (default), all channels are processed.
        **override :
            Keyword arguments forwarded to Channel.drift_corrected, e.g.
            ``points=100``. These override the stored drift parameters.

        Returns
        -------
        Test
            New Test instance with updated channels.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for drift correction.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.drift_corrected(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def filtered(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        their filtered versions (using Channel.filtered).

        Parameters
        ----------
        selector :
            Which channels to process (index, name, Channel, list, slice…).
            If None (default), all channels are processed.
        **override :
            Keyword arguments forwarded to Channel.filtered, e.g.
            ``btype="highpass"``, ``fc=0.5``, ``order=4``. These override
            the stored filter parameters.

        Returns
        -------
        Test
            New Test instance with updated channels.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for filtering.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.filtered(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def baseline_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        baseline-corrected versions (using Channel.baseline_corrected).

        Parameters
        ----------
        selector :
            Which channels to process (index, name, Channel, list, slice…).
            If None (default), all channels are processed.
        **override :
            Keyword arguments forwarded to Channel.baseline_corrected, e.g.
            ``type="linear"``. These override the stored baseline parameters.

        Returns
        -------
        Test
            New Test instance with updated channels.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for baseline correction.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.baseline_corrected(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> "Test":
        """
        Return a new Test where selected channels are replaced by
        trimmed versions (using Channel.trimmed).

        This is the generic manual-window trimming interface based on
        explicit ``t_start`` / ``t_end`` (in seconds).

        Parameters
        ----------
        selector :
            Which channels to process (index, name, Channel, list, slice…).
            If None (default), all channels are processed.
        **override :
            Keyword arguments forwarded to Channel.trimmed, typically
            including ``t_start`` and ``t_end`` (in seconds).

        Returns
        -------
        Test
            New Test instance with updated channels.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for trimming.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.trimmed(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_threshold(
        self,
        selector: ChannelSelector = None,
        ref: Optional[ChannelKey] = None,
        threshold: float = 0.01,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed using a single
        time window derived from one reference channel and a threshold
        criterion.

        Strategy
        --------
        - Choose a reference channel:
            * If `ref` is given, use that (must belong to this Test and
              be part of the selected set).
            * Otherwise, use the first selected channel.
        - On the reference channel, compute a threshold-based window via
          Channel.trim_by_threshold.
        - Extract (t_start, t_end) from the reference channel's trim_params.
        - Apply Channel.trimmed(t_start, t_end) to all selected channels.

        Parameters
        ----------
        selector :
            Which channels to trim (index, name, Channel, list, slice…).
            If None (default), all channels are trimmed.
        ref :
            Reference channel used to define the trim window. Can be an
            index, name, or Channel instance. If None, the first selected
            channel is used. The reference must be part of the selected set.
        threshold :
            Threshold value in signal units used to detect when the motion
            starts/stops (see Channel.trim_by_threshold).
        use_abs :
            If True, thresholding is applied to abs(signal). If False,
            thresholding is applied to the raw signal.
        buffer_before, buffer_after :
            Time buffers (in seconds) to extend the window before/after
            the detected start/end times.
        processed :
            Whether to use processed data for the reference channel when
            computing the window.
        use_cache :
            Whether to use the Channel-level processing cache for the
            reference channel.

        Returns
        -------
        Test
            New Test instance with aligned trimming across channels.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for threshold-based trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference Channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]  # int or str via __getitem__
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_threshold(
            threshold=threshold,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_fraction_of_peak(
        self,
        selector: ChannelSelector = None,
        ref: Optional[ChannelKey] = None,
        fraction: float = 0.05,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed to a single
        time window derived from a fraction-of-peak criterion on one
        reference channel.

        Strategy
        --------
        - Choose a reference channel (like trimmed_by_threshold).
        - On the reference channel, compute the window via
          Channel.trim_by_fraction_of_peak.
        - Extract (t_start, t_end) from the reference channel's trim_params.
        - Apply Channel.trimmed(t_start, t_end) to all selected channels.

        Parameters
        ----------
        selector :
            Which channels to trim (index, name, Channel, list, slice…).
            If None (default), all channels are trimmed.
        ref :
            Reference channel used to define the trim window. Can be an
            index, name, or Channel instance. If None, the first selected
            channel is used. The reference must be part of the selected set.
        fraction :
            Fraction of the peak amplitude in (0, 1] used to define the
            effective-motion window (see Channel.trim_by_fraction_of_peak).
        use_abs :
            If True, use absolute amplitude when computing the peak.
        buffer_before, buffer_after :
            Time buffers (in seconds) to extend the window before/after
            the detected start/end times.
        processed :
            Whether to use processed data for the reference channel when
            computing the window.
        use_cache :
            Whether to use the Channel-level processing cache for the
            reference channel.

        Returns
        -------
        Test
            New Test instance with aligned trimming across channels.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for fraction-of-peak trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference Channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_fraction_of_peak(
            fraction=fraction,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_arias(
        self,
        selector: ChannelSelector = None,
        ref: Optional[ChannelKey] = None,
        lower: float = 0.05,
        upper: float = 0.95,
        g: float = 9.81,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Test":
        """
        Return a new Test where selected channels are trimmed to a single
        Arias-intensity significant-duration window derived from one
        reference channel (e.g. 5–95% of Arias intensity).

        Strategy
        --------
        - Choose a reference channel (like trimmed_by_threshold).
        - On the reference channel, compute the Arias-based window via
          Channel.trim_by_arias.
        - Extract (t_start, t_end) from the reference channel's trim_params.
        - Apply Channel.trimmed(t_start, t_end) to all selected channels.

        Parameters
        ----------
        selector :
            Which channels to trim (index, name, Channel, list, slice…).
            If None (default), all channels are trimmed.
        ref :
            Reference channel used to define the trim window. Can be an
            index, name, or Channel instance. If None, the first selected
            channel is used. The reference must be part of the selected set.
        lower, upper :
            Lower and upper fractions of Arias intensity (in [0, 1]) that
            define the significant-duration window, typically 0.05 and 0.95.
        g :
            Gravitational acceleration used for Arias intensity, in m/s^2.
        buffer_before, buffer_after :
            Time buffers (in seconds) to extend the window before/after
            the detected lower/upper times.
        processed :
            Whether to use processed data for the reference channel when
            computing the window.
        use_cache :
            Whether to use the Channel-level processing cache for the
            reference channel.

        Returns
        -------
        Test
            New Test instance with aligned trimming across channels.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for Arias-based trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference Channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_arias(
            lower=lower,
            upper=upper,
            g=g,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Pairwise analysis
    # ------------------------------------------------------------------ #

    def cross_spectrum(
        self,
        x: ChannelKey,
        y: ChannelKey,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the cross power spectral density (CPSD) between two channels using ``scipy.signal.csd``.

        Parameters
        ----------
        x :
            Input (excitation) channel key (index, name or Channel instance).
        y :
            Output (response) channel key (index, name or Channel instance).
        processed : bool, optional
            If True (default), use processed data from each channel.
        use_cache : bool, optional
            If True (default), use the Channel-level processing cache.
        **kwargs :
            Additional keyword arguments forwarded to ``scipy.signal.csd``,
            e.g. ``nperseg``, ``window``, ``noverlap``.
            If ``nperseg`` is not given, a MATLAB-like default of ``min(256, n)`` is used.

        Returns
        -------
        f : np.ndarray
            Frequency array in Hz.
        Pxy : np.ndarray
            Complex cross-spectrum ``Pxy(f)``.
        """
        if isinstance(x, Channel):
            if x not in self.channels:
                raise ValueError("Input Channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if y not in self.channels:
                raise ValueError("Output Channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute cross spectrum of empty signal.")
        if x_data.size != y_data.size:
            raise ValueError("Channels must have the same length for cross spectrum.")
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Cross spectrum requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Cross spectrum requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        fs = 1.0 / ch_x.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, Pxy = sp.signal.csd(x_data, y_data, fs=fs, **kwargs)
        return f, Pxy

    def transfer_function(
        self,
        x: ChannelKey,
        y: ChannelKey,
        kind: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the frequency-domain transfer function between two channels.

        This uses cross- and auto-spectra computed with ``scipy.signal.csd``. Two standard estimators are supported:
        - H1: ``H1(f) = G_yx(f) / G_xx(f)``, preferred when the input is noisy.
        - H2: ``H2(f) = G_yy(f) / G_yx(f)``, preferred when the output is noisy.

        Here ``G_yx`` is the cross-spectrum between output ``y`` and input ``x``, and ``G_xx``, ``G_yy`` are the auto-spectra of input and output.

        Parameters
        ----------
        x :
            Input (excitation) channel key (index, name or Channel instance).
        y :
            Output (response) channel key (index, name or Channel instance).
        kind : {"H1", "H2"}, optional
            Type of transfer-function estimator (default "H1").
        processed : bool, optional
            If True (default), use processed data from each channel.
        use_cache : bool, optional
            If True (default), use the Channel-level processing cache.
        **kwargs :
            Additional keyword arguments forwarded to ``scipy.signal.csd``,
            e.g. ``nperseg``, ``window``, ``noverlap``.
            If ``nperseg`` is not given, a MATLAB-like default of ``min(256, n)`` is used.

        Returns
        -------
        f : np.ndarray
            Frequency array in Hz.
        H : np.ndarray
            Complex transfer function values ``H(f)``.
        """
        if isinstance(x, Channel):
            if x not in self.channels:
                raise ValueError("Input Channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if y not in self.channels:
                raise ValueError("Output Channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute transfer function of empty signal.")
        if x_data.size != y_data.size:
            raise ValueError(
                "Channels must have the same length for transfer function."
            )
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Transfer function requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Transfer function requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        fs = 1.0 / ch_x.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, Gyx = sp.signal.csd(y_data, x_data, fs=fs, **kwargs)
        _, Gxx = sp.signal.csd(x_data, x_data, fs=fs, **kwargs)
        _, Gyy = sp.signal.csd(y_data, y_data, fs=fs, **kwargs)
        kind_u = kind.upper()
        if kind_u == "H1":
            H = Gyx / Gxx
        elif kind_u == "H2":
            H = Gyy / Gyx
        else:
            raise ValueError(
                f"Unsupported transfer-function kind {kind!r}, use 'H1' or 'H2'."
            )
        return f, H

    def time_delay(
        self,
        x: ChannelKey,
        y: ChannelKey,
        processed: bool = True,
        use_cache: bool = True,
    ) -> float:
        """
        Estimate the time delay between two channels using cross-correlation.

        A positive delay means that the output ``y`` lags the input ``x``,
        based on the lag at which the cross-correlation between ``y`` and ``x`` is maximized.
        Parameters
        ----------
        x :
            Input (excitation) channel key (index, name or Channel instance).
        y :
            Output (response) channel key (index, name or Channel instance).
        processed : bool, optional
            If True (default), use processed data from each channel.
        use_cache : bool, optional
            If True (default), use the Channel-level processing cache.
        Returns
        -------
        tau : float
            Estimated time delay in seconds (positive if ``y`` lags ``x``).
        """
        if isinstance(x, Channel):
            if x not in self.channels:
                raise ValueError("Input Channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if y not in self.channels:
                raise ValueError("Output Channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute time delay for empty signal.")
        if x_data.size != y_data.size:
            raise ValueError(
                "Channels must have the same length for time-delay estimation."
            )
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Time-delay estimation requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Time-delay estimation requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        x0 = x_data - float(np.mean(x_data))
        y0 = y_data - float(np.mean(y_data))
        c = np.correlate(y0, x0, mode="full")
        lags = np.arange(-n + 1, n)
        k = lags[int(np.argmax(c))]
        return float(k * ch_x.dt)

    # ------------------------------------------------------------------ #
    # Basic modal identification (skeleton only)
    # ------------------------------------------------------------------ #

    def ema_model(
        self,
        input: ChannelKey,
        outputs: ChannelSelector,
        kind: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        **model_kwargs: Any,
    ):
        """
        Build and return an sdypy.EMA.Model for experimental modal analysis.

        This method computes FRFs between one input channel and multiple output channels,
        then constructs and returns an ``sdypy.EMA.Model`` instance using ``**model_kwargs``.

        The returned object provides pole estimation, stabilization charts,
        modal parameter extraction and FRF reconstruction.

        Parameters
        ----------
        input :
            Input (excitation) channel.
        outputs :
            Output (response) channels.
        kind : {"H1","H2"}, optional
            Transfer function estimator (default "H1").
        processed : bool, optional
            Use processed channel data (default True).
        use_cache : bool, optional
            Use Channel-level cache (default True).
        **model_kwargs :
            All additional keyword arguments are passed directly to
            ``sdypy.EMA.Model``. Typical options include:
                - ``lower``: lower frequency for pole estimation
                - ``upper``: upper frequency for pole estimation
                - ``pol_order_high``: highest model order for LSCF
                - ``driving_point``: index of driving FRF
                - ``frf_type``: "accelerance", "mobility", "receptance", ...

        Returns
        -------
        model : sdypy.EMA.Model

        Typical usage
        -------------
        After constructing the model with ``test.ema_model(...)``:

        1) Get poles (LSCF)::
               model.get_poles()
        2) Select stable poles (interactive or automatic)::
               model.select_poles()
              or
               model.select_closest_poles([f1, f2, ...])
        3) Print modal data (natural frequencies, damping and mode shapes)::
               acc.print_modal_data()
              or
               print(model.nat_freq)
               print(model.nat_xi)
               print(model.phi)
        4) Reconstruct FRFs and modal constants::
               frf_rec, modal_const = model.get_constants()
        """
        try:
            from sdypy import EMA
        except ImportError as exc:
            raise ImportError(
                "Experimental modal analysis requires the optional dependency "
                "'sdypy'. Install it with 'pip install sdypy-EMA'."
            ) from exc
        # Resolve channels
        if isinstance(input, Channel):
            ch_in = input
        else:
            ch_in = self[input]
        outs = list(self.iter_channels(outputs))
        if not outs:
            raise ValueError("No output channels selected.")
        # Compute FRFs for each output channel
        f_ref = None
        H_rows = []
        for ch_out in outs:
            f, H = self.transfer_function(
                x=ch_in,
                y=ch_out,
                kind=kind,
                processed=processed,
                use_cache=use_cache,
            )
            if f_ref is None:
                f_ref = f
            elif f.shape != f_ref.shape or not np.allclose(f, f_ref):
                raise ValueError("FRFs must share identical frequency grids.")
            H_rows.append(H)
        frf_matrix = np.vstack(H_rows)
        # Pass EVERYTHING to EMA.Model
        model = EMA.Model(
            frf_matrix,
            f_ref,
            **model_kwargs,
        )
        return model

    # ------------------------------------------------------------------ #
    # Multi-channel plotting
    # ------------------------------------------------------------------ #

    def _normalize_layout(self, layout: Any) -> list[list[Any]]:
        """
        Normalize a layout specification into a rectangular 2D list.

        Each cell will be:
        - None
        - a single ChannelKey / Channel
        - a sequence of ChannelKey / Channel
        """
        if not isinstance(layout, (list, tuple)):
            raise TypeError("layout must be a sequence of rows or cells.")
        if not layout:
            raise ValueError("layout must not be empty.")
        first = layout[0]
        if not isinstance(first, (list, tuple)):
            # 1D layout -> single row
            rows = [list(layout)]
        else:
            rows = [list(row) for row in layout]
        max_len = max(len(row) for row in rows)
        if max_len == 0:
            raise ValueError("layout rows must not be empty.")
        normalized: list[list[Any]] = []
        for row in rows:
            pad = max_len - len(row)
            if pad > 0:
                row = row + [None] * pad
            normalized.append(row)
        return normalized

    def _plot_one_channel(
        self,
        ch: Channel,
        ax: plt.Axes,
        plot_type: str,
        multi: bool,
        **kwargs: Any,
    ) -> None:
        """
        Internal helper to route plot_type to the appropriate Channel method.

        For multi-channel axes, use a generic kind label + legend.
        For single-channel axes, use the channel's axis label.
        """
        # Decide label behaviour
        if multi:
            # Generic kind on the y-axis, individual lines distinguished by legend
            include_label = False
            include_kind = True
            include_legend = True
        else:
            include_label = True
            include_kind = False
            include_legend = False
        pt = plot_type.lower()
        if pt in ("time", "timehistory", "time_history"):
            ch.plot(
                ax=ax,
                include_label=include_label,
                include_kind=include_kind,
                include_legend=include_legend,
                **kwargs,
            )
        elif pt in ("fourier", "fft"):
            ch.plot_fourier(
                ax=ax,
                **kwargs,
            )
            # Fourier plot uses its own labels; legend optional for multi
            if multi:
                # Add a legend entry using the line label that Channel.plot would use
                line_label = ch.label_legend or ch.name_user or ch.name_input
                if line_label:
                    for line in ax.get_lines():
                        if line.get_label() == "_nolegend_":
                            line.set_label(line_label)
                            break
                    ax.legend()
        elif pt in ("psd", "welch", "power"):
            ch.plot_psd(
                ax=ax,
                **kwargs,
            )
            if multi:
                line_label = ch.label_legend or ch.name_user or ch.name_input
                if line_label:
                    for line in ax.get_lines():
                        if line.get_label() == "_nolegend_":
                            line.set_label(line_label)
                            break
                    ax.legend()
        elif pt in ("arias", "husid"):
            ch.plot_arias(
                ax=ax,
                **kwargs,
            )
        elif pt in ("response", "response_spectrum", "rs"):
            ch.plot_response_spectrum(
                ax=ax,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown plot_type {plot_type!r}. "
                "Use 'timehistory', 'fourier', 'psd', 'arias', 'response', etc."
            )

    def plot_grid(
        self,
        layout: Any,
        plot_type: str = "timehistory",
        sharex: bool = True,
        sharey: bool = True,
        title_suffix: str | None = None,
        make_caption: bool = True,
        **kwargs: Any,
    ):
        """
        Plot channels from this Test in a grid of subplots.

        The layout argument describes how channels are arranged on the grid.
        Each cell in layout can be:
        - None: leave the subplot empty;
        - a single ChannelKey / Channel: one channel on that axes;
        - a sequence (tuple or list) of ChannelKey / Channel: multiple channels
          overlaid on the same axes, with a legend.

        Examples
        --------
        2x2 grid, one channel per subplot::
            test.plot_grid([[1, 2], [3, 4]])

        Ragged rows (padded with empty cell)::
            test.plot_grid([[1, 2], [3]])

        Multiple channels on one axes with legend::
            test.plot_grid([(2, 3), (4, 5)])

        Parameters
        ----------
        layout :
            Layout specification as described above (nested lists/tuples of
            ChannelKey or Channel, optionally with None cells).
        plot_type : str, optional
            Plot type: "timehistory" (time), "fourier", "psd", "arias",
            "response", etc. This is mapped to the corresponding Channel
            plotting method.
        sharex : bool, optional
            If True (default), subplots share the same x-axis.
        sharey : bool, optional
            If True (default), subplots share the same y-axis range.
        title_suffix : str or None, optional
            Optional suffix to append to the Test name for the figure title.
            The title is ``self.name`` if None, or ``f"{self.name}: {title_suffix}"``.
        make_caption : bool, optional
            If True, the function also returns a suggested figure caption string
            as a third return value.
        **kwargs :
            Additional keyword arguments forwarded to the underlying Channel
            plotting method (e.g. processed=False, fmax=..., etc).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created Figure.
        axes : numpy.ndarray
            2D array of Axes objects with shape (n_rows, n_cols).
        caption : str, optional
            If ``make_caption`` is True, a suggested figure caption is returned
            as a third element.
        """
        normalized = self._normalize_layout(layout)
        n_rows = len(normalized)
        n_cols = len(normalized[0])
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        all_channels: list[Channel] = []
        for i_row, row in enumerate(normalized):
            for j_col, cell in enumerate(row):
                ax = axes[i_row, j_col]
                if cell is None:
                    ax.set_visible(False)
                    continue
                if isinstance(cell, (list, tuple)):
                    keys = list(cell)
                else:
                    keys = [cell]
                channels: list[Channel] = []
                for key in keys:
                    if isinstance(key, Channel):
                        ch = key
                        if ch not in self.channels:
                            raise ValueError(
                                "Channel in layout is not part of this Test."
                            )
                    else:
                        ch = self[key]
                    channels.append(ch)
                    all_channels.append(ch)
                multi = len(channels) > 1
                for ch in channels:
                    self._plot_one_channel(
                        ch=ch,
                        ax=ax,
                        plot_type=plot_type,
                        multi=multi,
                        **kwargs,
                    )
        full_title = (
            self.name if title_suffix is None else f"{self.name}: {title_suffix}"
        )
        fig.suptitle(full_title)
        caption = ""
        if make_caption:
            uniq_channels: list[Channel] = []
            seen = set()
            for ch in all_channels:
                if id(ch) not in seen:
                    seen.add(id(ch))
                    uniq_channels.append(ch)
            channel_names = ", ".join(
                ch.name_user or ch.name_input or "<unnamed>" for ch in uniq_channels
            )
            caption = f"{full_title}. {plot_type} plots of channels: {channel_names}."
            return fig, axes, caption
        return fig, axes

    def plot_channels(
        self,
        selector: Any,
        ncols: int = 3,
        plot_type: str = "timehistory",
        sharex: bool = True,
        sharey: bool = True,
        title_suffix: str | None = None,
        make_caption: bool = False,
        **kwargs: Any,
    ):
        """
        Plot a list of channels from this Test in a grid with a fixed number of columns.

        Channels are selected via the given selector and arranged row-wise into
        subplots with ``ncols`` columns. This is convenient for plotting many
        similar channels (e.g. all accelerograms) at once.
        """
        channels = list(self.iter_channels(selector))
        if not channels:
            raise ValueError("No channels selected for plotting.")
        rows: list[list[Channel]] = []
        for i in range(0, len(channels), ncols):
            rows.append(channels[i : i + ncols])
        return self.plot_grid(
            rows,
            plot_type=plot_type,
            sharex=sharex,
            sharey=sharey,
            title_suffix=title_suffix,
            make_caption=make_caption,
            **kwargs,
        )
