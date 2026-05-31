import librosa
import numpy as np
import pyroomacoustics as pra
from scipy import signal
import pyroomacoustics as pra
import numpy as np
from scipy.signal import sosfilt, butter

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

# bande di ottava standard ISO 3382
bands = {
    125:  (88, 177),
    250:  (177, 354),
    500:  (354, 707),
    1000: (707, 1414),
    2000: (1414, 2828),
    4000: (2828, 5657),
}

def extract_ground_truth_rt60(impulse_response, sr):
    results = {}
    for fc, (low, high) in bands.items():
        filtered = bandpass_filter(impulse_response, low, high, sr)
        rt60 = pra.experimental.measure_rt60(filtered, sr)
        results[fc] = rt60
        print(f"{fc} Hz: RT60 = {rt60:.3f} s")
    
    # media ISO sulle bande 500 e 1000 Hz
    iso_mean = (results[500] + results[1000]) / 2
    print(f"Media ISO (500-1000 Hz): {iso_mean:.3f} s")
    return results, iso_mean

"""
def extract_ground_truth_rt60(impulse_response, sr):
     Estrae il RT60 dall'impulse responce usando il metodo di Schroeder
    Ritorna:
    rt60: stima del RT60 in secondi

    rt60 = pra.experimental.measure_rt60(impulse_response, sr)
    return rt60 """

