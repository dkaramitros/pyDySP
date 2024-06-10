import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import csd, welch
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar
from scipy import interpolate, optimize
from . import channel

class Test:
    
    def __init__(self):
        """
        Initialize Test instance with default values.
        """
        self.set_test_info(
            name="Default test",
            description="This is the default test.",
            filename="N/A",
            time="N/A",
            no_channels=0
        )
        self.channel = []

    def set_test_info(self, name: str = None, description: str = None, filename: str = None, time: str = None, no_channels: int = None) -> None:
        """
        Set test information.

        Parameters:
        name (str): Name of the test.
        description (str): Description of the test.
        filename (str): Filename of the test data.
        time (str): Time of the test.
        no_channels (int): Number of channels in the test.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if filename is not None:
            self.filename = filename
        if time is not None:
            self.time = time
        if no_channels is not None:
            self.no_channels = no_channels

    def add_channel(self) -> None:
        """
        Add a channel to the test.
        """
        self.channel.append(channel.Channel())
        if self.no_channels < len(self.channel):
            self.no_channels = len(self.channel)

    def set_channel_info(self, names: str = None, descriptions: str = None, units: str = None, calibrations: float = 1) -> None:
        """
        Set information for each channel.

        Parameters:
        names (str): List of names for channels.
        descriptions (str): List of descriptions for channels.
        units (str): List of units for channels.
        calibrations (float): Calibration factor for channels.
        """
        for i, channel in enumerate(self.channel):
            channel.set_channel_info(
                name=names[i],
                description=descriptions[i],
                unit=units[i],
                calibration=calibrations[i]
            )

    def get_test_info(self, print_info: bool = True):
        """
        Get the test information and optionally print it.

        Parameters:
        print_info (bool): If True, print the test information. Default is True.

        Returns:
        list: A list containing the test information.
        """
        # Gather the test information
        info = [
            self.name,
            self.description,
            self.filename,
            self.time,
            self.no_channels,
            [channel.name for channel in self.channel]
        ]
        # Print the test information if print_info is True
        if print_info:
            print(f"Name: {info[0]}")
            print(f"Description: {info[1]}")
            print(f"Filename: {info[2]}")
            print(f"Time: {info[3]}")
            print(f"Number of Channels: {info[4]}")
            print("Channel Names:")
            for idx, name in enumerate(info[5]):
                print(f"  {idx}: {name}")
        return info

    def read_equals(self, filename: str) -> None:
        """
        Read data from a .mat file and set test information and channel data accordingly.

        Parameters:
        filename (str): Path to the .mat file.
        
        Raises:
        FileNotFoundError: If the specified file does not exist.
        """
        try:
            imported_data = sp.io.loadmat(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found.")
        self.set_test_info(
            name=filename.split("/")[-1].split(".")[0],
            description="Project reference: " + imported_data['P_ref'][0],
            filename=imported_data['File_name'][0],
            time=imported_data['Testdate'][0] + imported_data['Time'][0],
            no_channels=imported_data['No_Channels'][0][0]
        )
        for i in range(self.no_channels):
            self.add_channel()
            self.channel[i].set_channel_data(
                raw_time=imported_data['t'].flatten(),
                raw_data=imported_data[f'chan{i+1}'].flatten()
            )
    
    def baseline_correct(self, **kwargs) -> None:
        """
        Apply baseline correction to each channel.

        Parameters:
        **kwargs: Additional keyword arguments to pass to the baseline method of each channel.
        """
        for channel in self.channel:
            channel.baseline_correct(**kwargs)

    def filter(self, **kwargs) -> None:
        """
        Apply a low-pass Butterworth filter to each channel.

        Parameters:
        **kwargs: Additional keyword arguments to pass to the filter method of each channel.
        """
        for channel in self.channel:
            channel.filter(**kwargs)
    
    def trim(self, **kwargs) -> None:
        """
        Trim the data for each channel.

        Parameters:
        **kwargs: Additional keyword arguments to pass to the trim method of each channel.
        """
        [start_0,end_0] = self.channel[0].trim(**kwargs)
        for kwarg in ["trim_method", "start", "end"]:
            if kwarg in kwargs:
                del kwargs[kwarg]
        for channel in self.channel[1:]:
            channel.trim(trim_method="Points", start=start_0, end=end_0, buffer=0, **kwargs)

    def plot(self, channels: np.ndarray = None, columns: int = 1, description: bool = False, **kwargs) -> plt.Axes:
        """
        Plot the data for specified channels.

        Parameters:
        channels (np.ndarray): Array of channel indices to plot.
        columns (int): Number of columns for subplots.
        description (bool): If True, includes channel description in plot.
        **kwargs: Additional keyword arguments to pass to the plot method of each channel.

        Returns:
        plt.Axes: The axes object containing the plots.
        """
        if channels is None:
            channels = np.arange(self.no_channels)
        no_channels = len(channels)
        rows = -(-no_channels // columns)
        figure, axes = plt.subplots(rows, columns, sharex=True, sharey=True)
        figure.suptitle(self.name)
        figure.set_tight_layout(True)
        for i, axis in enumerate(axes.flat):
            if i < no_channels:
                self.channel[channels[i]].plot(axis=axis, description=description, **kwargs)
        return axes
    
    def transfer_function(self, channel_from: int=0, channel_to: int=1, h_method: int=1, axis=None, xlim: float=50,
        find_peak: bool=True, find_damping: bool=True, f_min: float=0, f_max: float=50, **kwargs):
        """
        Compute and plot the transfer function between two channels, optionally finding the peak
        and damping within a specified frequency range.

        Parameters:
        channel_from (int): Index of the channel from which data is taken.
        channel_to (int): Index of the channel to which data is compared.
        h_method (int): Method to compute the transfer function (1 or 2).
        axis (matplotlib.axes._axes.Axes, optional): Axis on which to plot the transfer function.
        xlim (float): x-axis limit for the plot.
        find_peak (bool): Whether to find and mark the peak of the transfer function.
        find_damping (bool): Whether to find and mark the damping.
        f_min (float): Minimum frequency for the peak search range.
        f_max (float): Maximum frequency for the peak search range.
        **kwargs: Additional keyword arguments for signal processing functions.

        Returns:
        tuple: axis, transfer function data (frequencies and values), peak frequency and value, damping ratio
        """
        if axis is None:
            _, axis = plt.subplots()
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel(f"Transfer Function {self.channel[channel_to].name}/{self.channel[channel_from].name}")
        axis.set_xlim(0, xlim)
        axis.grid()
        # Compute transfer function
        fs = 1 / self.channel[channel_from]._timestep
        x_data = self.channel[channel_from]._data
        y_data = self.channel[channel_to]._data
        if h_method == 1:
            f, Pxy = sp.signal.csd(x=x_data, y=y_data, fs=fs, **kwargs)
            _, Pxx = sp.signal.welch(x=x_data, fs=fs, **kwargs)
            t = np.abs(Pxy / Pxx)
        else:
            f, Pyy = sp.signal.welch(x=y_data, fs=fs, **kwargs)
            _, Pxy = sp.signal.csd(x=x_data, y=y_data, fs=fs, **kwargs)
            t = np.abs(Pyy / Pxy)
        base_plot, = axis.plot(f, t, label=self.name)
        # Find peak within the specified range
        f_n, t_n, ksi = None, None, None
        if find_peak or find_damping:
            valid_indices = (f >= f_min) & (f <= f_max)
            if np.any(valid_indices):
                peak_index = np.argmax(t[valid_indices])
                peak_index = np.where(valid_indices)[0][peak_index]
                f_n, t_n = f[peak_index], t[peak_index]
                axis.plot(f_n, t_n, "o", color=base_plot.get_color())
        # Compute damping (half-bandwidth method)
        if find_damping and f_n is not None:
            try:
                t_hb = max(t_n / np.sqrt(2), t[0])
                eqn = sp.interpolate.interp1d(f, t - t_hb)
                f_1 = sp.optimize.root_scalar(eqn, bracket=[0, f_n], method='bisect').root
                f_2 = sp.optimize.root_scalar(eqn, bracket=[f_n, 2*f_n], method='bisect').root
                ksi = (f_2 - f_1) / (2 * f_n)
                axis.plot([f_1, f_2], [t_hb, t_hb], "--", color=base_plot.get_color())
            except ValueError:
                ksi = None
        return axis, [f, t], [f_n, t_n], ksi

    def export_to_csv(self, filename: str) -> None:
        """
        Export the data from all channels to a CSV file.

        Parameters:
        filename (str): Path to the output CSV file.

        Raises:
        ValueError: If there are no channels to export.
        """
        if self.no_channels == 0:
            raise ValueError("No channels available to export.")
        # Prepare header
        headers = ["Time"]
        for ch in self.channel:
            headers.append(ch.name)
        # Prepare data matrix
        max_points = max(len(ch._data) for ch in self.channel)
        data_matrix = np.empty((max_points, self.no_channels + 1))
        data_matrix[:] = np.nan
        # Fill time data
        data_matrix[:, 0] = self.channel[0]._time
        # Fill channel data
        for i, ch in enumerate(self.channel):
            data_matrix[:len(ch._data), i + 1] = ch._data
        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data_matrix)