# Import libraries
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class Test:
    
    def __init__(self):
        pass

    def set_test_info(self, name: str=None, description: str=None, filename: str=None, time: str=None, no_channels: int=None):
        if name != None: self.name = name
        if description != None: self.description = description
        if filename != None: self.filename = filename
        if time != None: self.time = time
        if no_channels != None: self.no_channels = no_channels

    def set_test_data(self, time: float=0, dataseries: float=0):
        self.data = [ Channel() for _ in range(int(self.no_channels))]
        for i,data in enumerate(self.data):
            data.set_channel_data(time=time, raw_data=dataseries[i])

    def set_channels_info(self, names: str=None, descriptions: str=None, units: str=None, calibrations: float=1):
        for i,channel in enumerate(self.channel):
            channel.set_channel_info(
                name = names[i],
                description = descriptions[i],
                unit = units[i],
                calibration = calibrations[i])

    def read_equals(self, filename):
        imported_data = sp.io.loadmat(filename)
        self.name = filename
        self.description = "Project reference: " + imported_data['P_ref'][0]
        self.filename = imported_data['File_name'][0]
        self.time = imported_data['Testdate'][0] + imported_data['Time'][0]
        self.no_channels = imported_data['No_Channels'][0][0]
        time = imported_data['t'].flatten()
        self.channel = [ Channel() for _ in range(int(self.no_channels)) ]
        for i,channel in enumerate(self.channel):
            channel.set_channel_data(raw_time = time, raw_data=imported_data[f'chan{i+1}'].flatten())
    
    def remove_offset(self, points: int=1000):
        for channel in self.channel:
            channel.remove_offset(points=points)
    
    def trim(self, start: int=None, end: int=None, ratio: float=0.05, max_threshold: float=0.01,
        buffer: int=100, time_shift: bool=True):
            [start_0,end_0] = self.channel[0].trim(start=start, end=end, ratio=ratio,
                max_threshold=max_threshold, buffer=buffer, time_shift=time_shift)  
            for channel in self.channel[1:]:
                channel.trim(start=start_0, end=end_0, time_shift=time_shift)

    def plot_timehistories(self, channels: int=None, columns: int=1):
        if channels == None: channels = np.arange(self.no_channels)
        channels = np.reshape(channels,[-1,columns])
        no_rows = np.shape(channels)[0]
        no_cols = np.shape(channels)[1]
        fig,axs = plt.subplots(no_rows, no_cols, sharex=True, sharey=True)
        fig.suptitle(self.name)
        fig.set_tight_layout(True)
        for row in range(no_rows):
            for col in range(no_cols):
                channel = self.channel[channels[row][col]]
                if no_cols > 1 and no_rows > 1:
                    channel.plot_timehistory(axs[row,col])
                else:
                    channel.plot_timehistory(axs[(row+1)*(col+1)-1])
        return axs

    def plot_fourier(self, channels: int=None, columns: int=3, xlim: float=50):
        if channels == None: channels = np.arange(self.no_channels)
        channels = np.reshape(channels,[-1,columns])
        no_rows = np.shape(channels)[0]
        no_cols = np.shape(channels)[1]
        fig,axs = plt.subplots(no_rows, no_cols, sharex=True, sharey=True)
        fig.suptitle(self.name)
        fig.set_tight_layout(True)
        for row in range(no_rows):
            for col in range(no_cols):
                channel = self.channel[channels[row][col]]
                if no_cols > 1 and no_rows > 1:
                    channel.plot_fourier(axs[row,col], xlim=xlim)
                else:
                    channel.plot_fourier(axs[(row+1)*(col+1)-1], xlim=xlim)
        return axs


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

    def trim(self, start: int=None, end: int=None, ratio: float=0.05, max_threshold: float=0.01,
        buffer: int=100, time_shift: bool=True):
        if start == None or end == None:
            min_threshold = ratio * np.amax(np.abs(self._data))
            max_threshold /= self.calibration
            threshold = min([min_threshold, max_threshold])
            start = max([np.argmax(np.abs(self._data) > threshold) - buffer, 0])
            end = np.size(self._data) - max([np.argmax(np.abs(np.flip(self._data)) > threshold) - buffer, 0]) 
        self._time = self._time[start:end]
        self._data = self._data[start:end]
        if time_shift == True:
            self._time -= self._time[0]
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
    
    def plot_timehistory(self, ax=None):
        if ax == None:
            fig, ax = plt.subplots()
        [t,y] = self.timehistory()
        ax.plot(t,y)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel(self.name + " (" + self.unit + ")")
        ax.grid()
        return ax

    def plot_fourier(self, ax=None, xlim: float=50):
        if ax == None:
            fig, ax = plt.subplots()
        [f,s] = self.fourier()
        ax.plot(f,s)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(self.name)
        ax.set_xlim(0,xlim)
        ax.grid()
        return ax