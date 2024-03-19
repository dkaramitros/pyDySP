# Import libraries
import scipy as sp
import numpy as np

class Test:
    
    def __init__(self):
        pass

    def set_test_info(self, name: str=None, description: str=None, filename: str=None, time: str=None, no_channels: int=0):
        if name != None: self.name = name
        if description != None: self.description = description
        if filename != None: self.filename = filename
        if time != None: self.time = time
        self.no_channels = no_channels

    def set_test_data(self, time: float=0, dataseries: float=0):
        self.data = [ Channel() for _ in range(int(self.no_channels))]
        for i,data in enumerate(self.data):
            data.set_channel_data(time=time, raw_data=dataseries[i])

    def set_channels_info(self, names: str=None, descriptions: str=None, units: str=None, calibrations: float=1):
        for i,channel_i in enumerate(self.channel):
            channel_i.set_channel_info(
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
        for i,channel_i in enumerate(self.channel):
            channel_i.set_channel_data(time = time, raw_data=imported_data[f'chan{i+1}'].flatten())
             

class Channel:
    
    def __init__(self):
        pass
    
    def set_channel_info(self, name: str="", description: str="", unit: str="", calibration: float=1):
        self.name = name
        self.description = description
        self.unit = unit
        self.calibration = calibration

    def set_channel_data(self, time: np.ndarray, raw_data: np.ndarray):
        self._time = time
        self._raw_data = raw_data
        self._data = raw_data
    
    def label(self):
        ylabel = self.name + " (" + self.unit + ")"
        return ylabel

    def timehistory(self):
        t = self._time
        y = self._data / self.calibration
        return np.array([t, y])
    
    def fourier(self):
        [t,y] = self.timehistory()
        no_t = np.size(t)
        dt = t[1] - t[0]
        no_f = int( 2**(no_t-1).bit_length() )
        spec = np.abs( np.fft.rfft(a=y, n=no_f) )
        f = np.fft.rfftfreq(n=no_f,d=dt)
        return np.array([f,spec])