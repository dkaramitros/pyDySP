# Import libraries
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class Test:
    
    def __init__(self):
        # Info
        self.name = "Default test"
        self.description = "This is the default test"
        self.filename = "N/A"
        self.time = "N/A"
        self.no_channels = 0
        # Data
        self.channel = []
        pass

    def set_test_info(self, name: str=None, description: str=None, filename: str=None, time: str=None, no_channels: int=None):
        if name != None: self.name = name
        if description != None: self.description = description
        if filename != None: self.filename = filename
        if time != None: self.time = time
        if no_channels != None: self.no_channels = no_channels

    def add_channel(self):
        self.no_channels += 1
        self.channel.append(Channel())

    def set_channel_info(self, names: str=None, descriptions: str=None, units: str=None, calibrations: float=1):
        for i,channel in enumerate(self.channel):
            channel.set_channel_info(
                name = names[i],
                description = descriptions[i],
                unit = units[i],
                calibration = calibrations[i])

    def read_equals(self, filename: str):
        imported_data = sp.io.loadmat(filename)
        self.set_test_info(
            name = filename.split("/")[-1].split(".")[0],
            description = "Project reference: " + imported_data['P_ref'][0],
            filename = imported_data['File_name'][0],
            time = imported_data['Testdate'][0] + imported_data['Time'][0],
            no_channels = imported_data['No_Channels'][0][0]
        )
        for i in range(self.no_channels):
            self.add_channel()
            self.channel[i].set_channel_data(
                raw_time = imported_data['t'].flatten(),
                raw_data = imported_data[f'chan{i+1}'].flatten()
            )
    
    def remove_offset(self, points: int=1000):
        for channel in self.channel:
            channel.remove_offset(points=points)
    
    def trim_threshold(self, **kwargs):
            [start_0,end_0] = self.channel[0].trim_threshold(**kwargs)
            for channel in self.channel[1:]:
                channel.trim_points(start=start_0, end=end_0, **kwargs)

    def plot(self, channels: np.ndarray = None, columns: int = 1, plot_type: str="Timehistory", **kwargs):
        if channels is None:
            channels = np.arange(self.no_channels)
        no_channels = len(channels)
        rows = -(-no_channels // columns)
        figure, axes = plt.subplots(rows, columns, sharex=True, sharey=True)
        figure.suptitle(self.name)
        figure.set_tight_layout(True)
        for i, axis in enumerate(axes.flat):
            if i < no_channels:
                self.channel[channels[i]].plot(axis=axis, plot_type=plot_type, **kwargs)
        return axes


class Channel:
    
    def __init__(self):
        pass
    
    def set_channel_info(self, name: str="", description: str="", unit: str="", calibration: float=1):
        self.name = name
        self.description = description
        self.unit = unit
        self.calibration = calibration

    def set_channel_data(self, raw_time: np.ndarray, raw_data: np.ndarray):
        self._raw_time = raw_time
        self._time = raw_time
        self._raw_data = raw_data
        self._data = raw_data

    def reset_raw_data(self):
        self._time = self._raw_time
        self._data = self._raw_data

    def remove_offset(self, points: int=1000):
        self._data = self._raw_data - np.average(self._raw_data[:points])

    def trim_points(self, start: int=None, end: int=None, buffer: int=100, time_shift: bool=True):
        start = max([start - buffer, 0])
        end = min([end + buffer, np.size(self._time)])
        self._time = self._time[start:end]
        self._data = self._data[start:end]
        if time_shift == True:
            self._time -= self._time[0]
        return [start,end]

    def trim_threshold(self, ratio: float=0.05, max_threshold: float=0.01, **kwargs):
        threshold = min([
            ratio * np.amax(np.abs(self._data)),
            max_threshold / self.calibration
        ])
        start = np.argmax(np.abs(self._data) > threshold)
        end = np.size(self._data) - np.argmax(np.abs(np.flip(self._data)) > threshold)
        self.trim_points(start=start, end=end, **kwargs)
        return [start,end]

    def timehistory(self):
        t = self._time
        y = self._data / self.calibration
        return np.array([t, y])
    
    def fourier(self):
        [t,y] = self.timehistory()
        no_t = np.size(t)
        dt = t[1] - t[0]
        no_f = int( 2**(no_t-1).bit_length() )
        s = np.abs( np.fft.rfft(a=y, n=no_f) )
        f = np.fft.rfftfreq(n=no_f,d=dt)
        return np.array([f,s])

    def plot(self, plot_type: str="Timehistory", axis=None, **kwargs):
        if axis == None:
            figure, axis = plt.subplots()
        match plot_type:
            case "Timehistory":
                [x,y] = self.timehistory()
                axis.set_xlabel("Time (sec)")
                axis.set_ylabel(self.name + " (" + self.unit + ")")
            case "Fourier":
                [x,y] = self.fourier()
                axis.set_xlabel("Frequency (Hz)")
                axis.set_ylabel(self.name)
                if "xlim" in kwargs:
                    xlim = kwargs["xlim"]
                else:
                    xlim = 50
                axis.set_xlim(0,xlim)
        axis.plot(x,y)
        axis.grid()
        return axis