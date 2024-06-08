import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.integrate import cumulative_trapezoid

class Channel:
    
    def __init__(self):
        self.set_channel_info(
            name = "Default channel",
            description = "This is the default channel.",
            unit = "Undefined unit",
            calibration = 1
        )
        self.set_channel_data(
            raw_time = np.zeros(2),
            raw_data = np.zeros(2)
        )
    
    def set_channel_info(self, name: str=None, description: str=None, unit: str=None, calibration: float=None):
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if unit is not None:
            self.unit = unit
        if calibration is not None:
            self.calibration = calibration

    def set_channel_data(self, raw_time: np.ndarray, raw_data: np.ndarray):
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

    def reset_raw_data(self):
        self._time = self._raw_time
        self._data = self._raw_data
        self._points = self._raw_points
        self._timestep = self._raw_timestep

    def baseline(self, **kwargs):
        self._data = detrend(self._raw_data, **kwargs)

    def filter(self, order: int=2, cutoff: float=50):
        b, a = butter(N=order, Wn=cutoff, btype='low', fs=1/self._timestep)
        self._data = filtfilt(b, a, self._data)

    def trim(self, buffer: int=100, time_shift: bool=True, trim_method: str="Threshold",
             start: int=0, end: int=0, threshold_ratio: float=0.05, threshold_acc: float=0.01):
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

    def timehistory(self):
        t = self._time
        y = self._data / self.calibration
        index = np.argmax(np.abs(y))
        t_max = t[index]
        y_max = y[index]
        return np.array([t, y]), [t_max, y_max]
    
    def fourier(self):
        [t, y] = self.timehistory()[0]
        _no_freqs = int(2 ** (self._points - 1).bit_length())
        f = np.fft.rfftfreq(n=_no_freqs, d=self._timestep)
        s = np.abs(np.fft.rfft(a=y, n=_no_freqs))
        index = np.argmax(f)
        f_n = f[index]
        s_max = s[index]
        return np.array([f, s]), [f_n, s_max]

    def welch(self, **kwargs):
        f, p = welch(x=self._data, fs=1/self._timestep, **kwargs)
        index = np.argmax(f)
        f_n = f[index]
        p_max = p[index]
        return np.array([f, p]), [f_n, p_max]

    def arias(self, g: float=9.81):
        arias = cumulative_trapezoid(
            x=self._time,
            y=np.pi / 2 / 9.81 * (g * self._data / self.calibration) ** 2
        )
        arias = np.append(arias, arias[-1])
        start = np.argmax(arias > 0.05 * arias[-1])
        end = np.argmax(arias > 0.95 * arias[-1])
        duration = self._time[end] - self._time[start]
        return [self._time, arias], arias[-1], duration, [start, end]

    def plot(self, plot_type: str="Timehistory", name: bool=True, description: bool=True, axis=None, **kwargs):
        if axis is None:
            figure, axis = plt.subplots()
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
        if freq_plot:
            xlim = kwargs.get("xlim", 50)
            axis.set_xlim(0, xlim)
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