# Import libraries
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class Test:
    
    def __init__(self):
        self.set_test_info(
            name = "Default test",
            description = "This is the default test.",
            filename = "N/A",
            time = "N/A",
            no_channels = 0
        )
        self.channel = []

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
    
    def baseline(self, **kwargs):
        for channel in self.channel:
            channel.baseline(**kwargs)

    def filter(self, **kwargs):
        for channel in self.channel:
            channel.filter(**kwargs)
    
    def trim(self, **kwargs):
            [start_0,end_0] = self.channel[0].trim(**kwargs)
            for kwarg in ["trim_method", "start", "end"]:
                if kwarg in kwargs:
                    del kwargs[kwarg]
            for channel in self.channel[1:]:
                channel.trim(trim_method="Points", start=start_0, end=end_0, **kwargs)

    def plot(self, channels: np.ndarray=None, columns: int=1, description: bool=False, **kwargs):
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
    
    def transfer(self, channel_from: int=0, channel_to: int=1, h_method: int=1, axis=None,
        find_peak: bool=True, find_damping: bool=True, xlim: float=50, **kwargs):
        # Plot
        if axis == None:
            figure, axis = plt.subplots()
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Transfer Function "+self.channel[channel_to].name+"/"+self.channel[channel_from].name)
        axis.set_xlim(0,xlim)
        axis.grid()
        # Transfer function
        if h_method == 1:
            [f,Pxy] = sp.signal.csd(x=self.channel[channel_from]._data, y=self.channel[channel_to]._data,
                fs=1/self.channel[channel_from]._timestep, **kwargs)
            [_,Pxx] = sp.signal.welch(x=self.channel[channel_from]._data,
                fs=1/self.channel[channel_from]._timestep, **kwargs)
            t = np.abs(Pxy / Pxx)
        else:
            [f,Pyy] = sp.signal.welch(x=self.channel[channel_to]._data,
                fs=1/self.channel[channel_from]._timestep, **kwargs)
            [_,Pxy] = sp.signal.csd(x=self.channel[channel_from]._data, y=self.channel[channel_to]._data,
                fs=1/self.channel[channel_from]._timestep, **kwargs)
            t = np.abs(Pyy / Pxy)
        base_plot, = axis.plot(f,t,label=self.name)
        # Peak
        if find_peak or find_damping:
            index_n = np.argmax(t)
            f_n = f[index_n]
            t_n = t[index_n]
            axis.plot(f_n, t_n, "o", color=base_plot.get_color())
        # Damping (half-bandwidth method)
        if find_damping:
            t_hb = max(t_n/np.sqrt(2),t[0])
            eqn = sp.interpolate.interp1d(f,t-t_hb)
            f_1 = sp.optimize.root_scalar(eqn, bracket=[0,f_n], method='bisect').root
            f_2 = sp.optimize.root_scalar(eqn, bracket=[f_n,2*f_n], method='bisect').root
            ksi = (f_2-f_1) / (2*f_n)
            axis.plot([f_1,f_2], [t_hb,t_hb], "--", color=base_plot.get_color())
        return axis,[f,t],[f_n,t_n],ksi


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
        if name != None: self.name = name
        if description != None: self.description = description
        if unit != None: self.unit = unit
        if calibration != None: self.calibration = calibration

    def set_channel_data(self, raw_time: np.ndarray, raw_data: np.ndarray):
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
        self._data = sp.signal.detrend(self._raw_data, **kwargs)

    def filter(self, order: int=2, cutoff: float=50):
        b, a = sp.signal.butter(N=order, Wn=cutoff, btype='low', fs=1/self._timestep)
        self._data = sp.signal.filtfilt(b, a, self._data)

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
                [start,end] = self.arias[3]
        start = max([start - buffer, 0])
        end = min([end + buffer, np.size(self._time)])
        self._time = self._time[start:end]
        self._data = self._data[start:end]
        self._points = np.size(self._time)
        if time_shift == True:
            self._time -= self._time[0]
        return [start,end]

    def timehistory(self):
        t = self._time
        y = self._data / self.calibration
        index = np.argmax(np.abs(y))
        t_max = t[index]
        y_max = y[index]
        return np.array([t,y]), [t_max,y_max]
    
    def fourier(self):
        [t,y] = self.timehistory()[0]
        _no_freqs = int( 2**(self._points-1).bit_length() )
        f = np.fft.rfftfreq(n=_no_freqs, d=self._timestep)
        s = np.abs( np.fft.rfft(a=y, n=_no_freqs) )
        index = np.argmax(f)
        f_n = f[index]
        s_max = s[index]
        return np.array([f,s]), [f_n,s_max]

    def welch(self, **kwargs):
        [f,p] = sp.signal.welch(x=self._data, fs=1/self._timestep, **kwargs)
        index = np.argmax(f)
        f_n = f[index]
        p_max = p[index]
        return np.array([f,p]), [f_n,p_max]

    def arias(self, g: float=9.81):
        arias = sp.integrate.cumulative_trapezoid(
            x=self._time,
            y=np.pi/2/9.81 * (g * self._data/self.calibration)**2
        )
        arias = np.append(arias,arias[-1])
        start = np.argmax(arias > 0.05*arias[-1])
        end = np.argmax(arias > 0.95*arias[-1])
        duration = self._time[end] - self._time[start]
        return [self._time,arias], arias[-1], duration, [start,end]

    def plot(self, plot_type: str="Timehistory", name: bool=True, description: bool=True, axis=None, **kwargs):
        if axis == None:
            figure, axis = plt.subplots()
        freq_plot = False
        match plot_type:
            case "Timehistory":
                [x,y] = self.timehistory()[0]
                xlabel = "Time (sec)"
                ydesc = "Timehistory (" + self.unit + ")"
            case "Fourier":
                [x,y] = self.fourier()[0]
                xlabel = "Frequency (Hz)"
                ydesc = "Fourier Amplitude"
                freq_plot = True
            case "Power":
                [x,y] = self.welch(**kwargs)[0]
                xlabel = "Frequency (Hz)"
                ydesc = "Power Spectral Density"
                freq_plot = True
            case "Arias":
                [x,y] = self.arias()[0]
                xlabel = "Time (sec)"
                ydesc = "Arias Intensity (m/s)"
        if freq_plot:
            if "xlim" in kwargs:
                xlim = kwargs["xlim"]
            else:
                xlim = 50
            axis.set_xlim(0,xlim)        
        axis.plot(x,y)
        ylabel = ""
        if name:
            ylabel += self.name
        if description:
            ylabel +=  " " + ydesc
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid()
        return axis