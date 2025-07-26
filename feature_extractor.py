# This code extracts the primary features from raw ECG signals collected from AD8232 sensor

#Importing the necessary libraries
import time
import serial
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt 

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

def bandpass_filter(signal, lowcut = 0.5, upcut = 20, fs = 50, order = 2):
    """
    Applies bandpass filter on the raw ECG signals, allowing signals within 0.5-40Hz to pass through
    Returns the filtered signal
    """
    nyq = 0.5*fs
    low = lowcut/nyq
    high = upcut/nyq
    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered

def detect_r_peaks(ecg_filtered, fs=50):
    """
    Implementing R-peak detection from the ECG signal
    Returns the indices of detected R-peaks
    """
    ecg_normalised = (ecg_filtered - np.mean(ecg_filtered))/np.std(ecg_filtered)
    ecg_sqrr = ecg_normalised ** 2
    ma_window = int(0.150*fs)
    ma = np.convolve(ecg_sqrr, np.ones(ma_window)/ma_window, mode="same")
    r_peaks, _ = find_peaks(ma, distance=0.6*fs, height=np.mean(ma))
    return r_peaks

def extract_features(ecg_filtered, r_peaks, fs=50):
    """
    Extracts 11 features(primary_features) from the raw signals
    primary_features = ['0_pre-RR','0_post-RR','0_pPeak','0_tPeak','0_rPeak','0_sPeak','0_qPeak','0_qrs_interval','0_pq_interval','0_qt_interval','0_st_interval']
    Returns numpy array of the extracted features
    """
    features = []
    n = len(r_peaks)
    for i in range(1, n-1):
        pre_RR = (r_peaks[i] - r_peaks[i-1])/fs
        post_RR = (r_peaks[i+1] - r_peaks[i])/fs

        segment = ecg_filtered[r_peaks[i]-int(0.3*fs): r_peaks[i]+int(0.4*fs)]
        curr_r_index = int(0.3*fs)

        q_peak = curr_r_index - np.argmin(segment[:curr_r_index])
        s_peak = curr_r_index + np.argmin(segment[curr_r_index:])

        p_search_region = segment[q_peak - int(0.15*fs):q_peak]
        t_search_region = segment[s_peak:s_peak + int(0.2*fs)]

        p_peak = q_peak - np.argmin(p_search_region) if len(p_search_region) > 0 else q_peak
        t_peak = s_peak + np.argmax(t_search_region) if len(t_search_region) > 0 else t_peak

        r_peak_time = r_peaks[i] / fs
        q_peak_time = (r_peaks[i] - (curr_r_index - q_peak)) / fs
        s_peak_time = (r_peaks[i] + (s_peak - curr_r_index)) / fs
        p_peak_time = (r_peaks[i] - (curr_r_index - p_peak)) / fs
        t_peak_time = (r_peaks[i] + (t_peak - curr_r_index)) / fs

        qrs_interval = s_peak_time - q_peak_time
        pq_interval = r_peak_time - p_peak_time
        qt_interval = t_peak_time - q_peak_time
        st_interval = t_peak_time - s_peak_time

        features = [
            pre_RR,
            post_RR,
            p_peak_time,
            t_peak_time,
            r_peak_time,
            s_peak_time,
            q_peak_time,
            qrs_interval,
            pq_interval,
            qt_interval,
            st_interval
        ]
        
        return features

def plot_data(ecg_raw, ecg_filtered):
    """
    Plots the extracted raw ECG signals and the filtered ECG signals
    """
    plt.plot(ecg_raw, label="Raw ECG Signal")
    plt.plot(ecg_filtered, label="Filtered ECG Signal")
    plt.legend()
    plt.title("ECG Signal - Raw vs Filtered")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


ecg_raw = [1890, 1009, 817, 726, 778, 755, 743, 720, 653, 576, 594, 637, 656, 654,
           620, 601, 592, 570, 559, 545, 573, 579, 706, 607, 601, 621, 662, 489,
           1733, 2119, 837, 677, 702, 711, 771, 952, 1033, 1090, 1077, 1153, 1406, 1504,
           1513, 1684, 1759, 1731, 1804, 1823, 1856, 1943, 2011, 2027, 2110, 2058, 2078, 2113,
           2139, 1967, 3115, 4095, 2209, 2101, 2111, 2132, 2161, 2195, 2202, 2180, 2157, 2185,
           2275, 2324, 2340, 2341, 2301, 2301, 2303, 2311, 2289, 2322, 2322, 2347, 2327, 2418,
           2353, 2336, 2306, 2305, 2199, 2737, 4095, 2195, 2192, 2186, 2201, 2195, 2225, 2251,
           2224, 2177, 2198, 2253, 2299, 2294, 2294, 2304, 2309, 2307, 2300, 2294, 2297, 2283,
           2325, 2317, 2409, 2305, 2257, 2256, 2246, 2177, 2453, 4095, 2382, 2154, 2160, 2177,
           2181, 2188, 2174, 2128, 2102, 2074, 2129, 2160, 2174, 2177, 2166, 2163, 2161, 2160,
           2154, 2141, 2154, 2146, 2208, 2163, 2102, 2096, 2097, 2074, 1973, 4095, 2447, 1993,
           1955, 1956, 1969, 1990, 1985, 1968, 1925, 1878, 1933, 2011, 2031, 2047, 2046, 2082,
           2101, 2112, 2117, 2115, 2116, 2127, 2155, 2163, 2225, 2145, 2083, 2079, 2080, 2044,
           2096, 4095, 2167, 1946, 1907, 1917, 1920, 1934, 1914, 1885, 1833, 1808, 1871, 1921,
           1933, 1918, 1931, 1906, 1903, 1884, 1872, 1872, 1861, 1872, 1907, 1915, 1995, 1913,
           1873, 1878, 1865, 1950, 2135, 4095, 2311, 2061, 2000, 2005, 1994, 1977, 1958, 1915,
           1849, 1807, 1857, 1936, 1939, 1939, 1937, 1936, 1929, 1920, 1899, 1903, 1889, 1886,
           1904, 1921, 2025, 1923, 1867, 1867, 1872, 1834, 1963, 4011, 2127, 1763]
# ecg_raw = read_data_from_serial()
ecg_filtered = bandpass_filter(ecg_raw)
r_peaks = detect_r_peaks(ecg_filtered)
features = extract_features(ecg_filtered, r_peaks)
print(ecg_raw)
print(ecg_filtered)
print(r_peaks)
print(features)