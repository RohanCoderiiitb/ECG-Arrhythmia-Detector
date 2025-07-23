# This code extracts the primary features from raw ECG signals collected from AD8232 sensor

#Importing the necessary libraries
import time
import serial
import numpy as np
from scipy.signal import butter, filtfilt

def read_data_from_serial(port = 'com4', baud_rate = 115200, duration = 5, sampling_rate=50):
    """
    Reads the raw ECG signals from AD8232 from COM4
    Returns a numpy array of the signals read
    """
    ser = serial.Serial(port, baudrate=baud_rate)
    time.sleep(2)
    ecg = []
    no_of_samples = duration * sampling_rate
    while len(ecg) < no_of_samples:
        try:
            line = ser.readline().decode().strip()
            if line.isdigit():
                ecg.append(int(line))
        except:
            continue
    ser.close()
    return np.array(ecg)

def bandpass_filter(signal, lowcut = 0.5, upcut = 40, fs = 50, order = 2):
    """
    Applies bandpass filter on the raw ECG signals, allowing signals within 0.5-40Hz to pass through
    Returns the filtered signal
    """
    nyq = 0.5*fs
    low = lowcut/nyq
    high = upcut/nyq
    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(a, b, signal)
    return filtered


ecg_raw = read_data_from_serial()
ecg_filtered = bandpass_filter(ecg_raw)