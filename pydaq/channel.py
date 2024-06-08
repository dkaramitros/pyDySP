import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.integrate import cumulative_trapezoid

class Channel:
    
    def __init__(self):
        """
        Initializes the Channel instance with default values.
        """
        self.set_channel_info(
            name="Default channel",
            description="This is the default channel.",
            unit="Undefined unit",
            calibration=1
        )
        self.set_channel_data(
            raw_time=np.zeros(2),
            raw_data=np.zeros(2)
        )
    
    def set_channel_info(self, name: str = None, description: str = None, unit: str = None, calibration: float = None) -> None:
        """
        Sets the channel information.

        Parameters:
        name (str): Name of the channel.
        description (str): Description of the channel.
        unit (str): Unit of measurement for the data.
        calibration (float): Calibration factor for the data.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if unit is not None:
            self.unit = unit
        if calibration is not None:
            self.calibration = calibration

    def set_channel_data(self, raw_time: np.ndarray, raw_data: np.ndarray) -> None:
        """
        Sets the raw time and data for the channel.

        Parameters:
        raw_time (np.ndarray): Array of time values.
        raw_data (np.ndarray): Array of data values.

        Raises:
        ValueError: If raw_time and raw_data have different shapes or contain less than two elements.
        """
        if raw_time.shape != raw_data.shape:
            raise ValueError("raw_time and raw_data must have the same shape")
        if len(raw_time) < 2:
            raise ValueError("raw_time and raw_data must contain at least two elements")
        self._raw_time = raw_time
        self._raw_data = raw_data
        self._raw_points = np.size(self._raw_data)
        self._raw_timestep = self._raw_time[1] - self._raw_time[0]
        self._time = raw_time
        self._data = raw_data
        self._points = np.size(self._time)
        self._timestep = self._time[1] - self._time[0]

    def reset_raw_data(self) -> None:
        """
        Resets the processed data to the raw data.
        """
        self._time = self._raw_time
        self._data = self._raw_data
        self._points = self._raw_points
        self._timestep = self._raw_timestep

    def baseline(self, **kwargs) -> None:
        """
        Removes the linear trend from the raw data using scipy.signal.detrend.

        Parameters:
        **kwargs: Additional keyword arguments to pass to scipy.signal.detrend.
        """
        self._data = detrend(self._raw_data, **kwargs)

    def filter(self, order: int = 2, cutoff: float = 50) -> None:
        """
        Applies a low-pass Butterworth filter to the data.

        Parameters:
        order (int): The order of the filter.
        cutoff (float): The cutoff frequency of the filter.
        """
        b, a = butter(N=order, Wn=cutoff, btype='low', fs=1/self._timestep)
        self._data = filtfilt(b, a, self._data)

    def trim(self, buffer: int = 100, time_shift: bool = True, trim_method: str = "Threshold",
             start: int = 0, end: int = 0, threshold_ratio: float = 0.05, threshold_acc: float = 0.01) -> list[int]:
        """
        Trims the data based on the specified method.

        Parameters:
        buffer (int): Number of points to include as buffer around the trimmed data.
        time_shift (bool): If True, shifts the time axis to start at zero.
        trim_method (str): Method to use for trimming ('Points', 'Threshold', 'Arias').
        start (int): Starting index for 'Points' method.
        end (int): Ending index for 'Points' method.
        threshold_ratio (float): Ratio threshold for 'Threshold' method.
        threshold_acc (float): Acceleration threshold for 'Threshold' method.

        Returns:
        list: The start and end indices used for trimming.

        Raises:
        ValueError: If an unknown trim_method is specified.
        """
        if self._points < self._raw_points:
            self.reset_raw_data()
        match trim_method:
            case "Points":
                pass
            case "Threshold":
                threshold = min([
                    threshold_ratio * np.amax(np.abs(self._data)),
                    threshold_acc / self.calibration
                ])
                start = np.argmax(np.abs(self._data) > threshold)
                end = np.size(self._data) - np.argmax(np.abs(np.flip(self._data)) > threshold)
            case "Arias":
                [start, end] = self.arias()[3]
            case _:
                raise ValueError(f"Unknown trim_method: {trim_method}")
        start = max([start - buffer, 0])
        end = min([end + buffer, np.size(self._time)])
        self._time = self._time[start:end]
        self._data = self._data[start:end]
        self._points = np.size(self._time)
        if time_shift:
            self._time -= self._time[0]
        return [start, end]

    def timehistory(self) -> tuple[np.ndarray, list[float]]:
        """
        Returns the time history data.

        Returns:
        np.ndarray: Array of time and scaled data values.
        list: Maximum time and data values.
        """
        t = self._time
        y = self._data / self.calibration
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
        index = np.argmax(f)
        f_n = f[index]
        s_max = s[index]
        return np.array([f, s]), [f_n, s_max]

    def welch(self, **kwargs) -> tuple[np.ndarray, list[float]]:
        """
        Computes the Power Spectral Density using Welch's method.

        Parameters:
        **kwargs: Additional keyword arguments to pass to scipy.signal.welch.

        Returns:
        np.ndarray: Array of frequencies and power spectral densities.
        list: Maximum frequency and power spectral density values.
        """
        f, p = welch(x=self._data, fs=1/self._timestep, **kwargs)
        index = np.argmax(f)
        f_n = f[index]
        p_max = p[index]
        return np.array([f, p]), [f_n, p_max]

    def arias(self, g: float = 9.81) -> tuple[list[np.ndarray, np.ndarray], float, float, list[int]]:
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
            x=self._time,
            y=np.pi / 2 / 9.81 * (g * self._data / self.calibration) ** 2
        )
        start = np.argmax(arias > 0.05 * arias[-1])
        end = np.argmax(arias > 0.95 * arias[-1])
        duration = self._time[end] - self._time[start]
        return [self._time, arias], arias[-1], duration, [start, end]

    def plot(self, plot_type: str = "Timehistory", name: bool = True, description: bool = True, axis=None, **kwargs) -> plt.Axes:
        """
        Plots the specified type of data.

        Parameters:
        plot_type (str): Type of plot ('Timehistory', 'Fourier', 'Power', 'Arias').
        name (bool): If True, includes the channel name in the ylabel.
        description (bool): If True, includes the data description in the ylabel.
        axis: Matplotlib axis to plot on. If None, creates a new axis.
        **kwargs: Additional keyword arguments for the plot.

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
                ydesc = "Timehistory (" + self.unit + ")"
            case "Fourier":
                [x, y] = self.fourier()[0]
                xlabel = "Frequency (Hz)"
                ydesc = "Fourier Amplitude"
                freq_plot = True
            case "Power":
                [x, y] = self.welch(**kwargs)[0]
                xlabel = "Frequency (Hz)"
                ydesc = "Power Spectral Density"
                freq_plot = True
            case "Arias":
                [x, y] = self.arias()[0]
                xlabel = "Time (sec)"
                ydesc = "Arias Intensity (m/s)"
            case _:
                raise ValueError(f"Unknown plot_type: {plot_type}")
        if freq_plot:
            axis.set_xlim(0, kwargs.get("xlim", 50))
        axis.plot(x, y)
        ylabel = ""
        if name:
            ylabel += self.name
        if description:
            ylabel += " " + ydesc
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid()
        return axis