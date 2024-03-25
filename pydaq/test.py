# Import libraries
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from . import channel

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
        self.channel.append(channel.Channel())

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