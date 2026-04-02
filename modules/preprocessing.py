# modules/preprocessing.py

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import noisereduce as nr

def load_audio(path, config):
    """
    Carica e ricampiona il file audio.
    offset e duration vengono dalla GUI tramite config.
    Se duration è 0 o None, carica tutto il file.
    """
    duration = config.get('duration', None)
    if duration == 0:
        duration = None  # librosa interpreta None come "tutto il file"

    y, sr = librosa.load(path, sr=config.get('target_sr', 44100), offset=config.get('offset', 0.0), duration=duration)
    return y, sr

def estimate_noise_profile(y, sr, noise_duration=2.0):
    """
    Estima il profilo del rumore dai primi N secondi.
    ATTENZIONE: funziona solo se quei secondi sono silenzio puro.
    Verificare sempre a mano prima di usare.
    """
    n_samples = int(sr * noise_duration)
    return y[:n_samples]

def remove_noise(y, sr, noise_profile, prop_decrease=0.7):
    """Sottrazione spettrale del rumore."""
    return nr.reduce_noise(y=y, sr=sr, 
                          y_noise=noise_profile,
                          prop_decrease=prop_decrease)

def bandpass_filter(y, sr, lowcut=40, highcut=16000, order=5):
    """Filtro passa-banda Butterworth."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, y)

def preprocess(path, config):
    """
    Pipeline completa di preprocessing.
    config è un dizionario con tutti i parametri.
    """
    y, sr = load_audio(path, 
                       target_sr=config['target_sr'],
                       offset=config.get('offset', 0),
                       duration=config.get('duration', None))
    
    noise = estimate_noise_profile(y, sr, 
                                   config.get('noise_duration', 2.0))
    y_clean = remove_noise(y, sr, noise, 
                          config.get('prop_decrease', 0.7))
    y_filtered = bandpass_filter(y_clean, sr,
                                 config.get('lowcut', 40),
                                 config.get('highcut', 16000))
    return y_filtered, sr