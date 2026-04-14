import librosa
import numpy as np
import pyroomacoustics as pra

def extract_ground_truth_rt60(impulse_responce, sr):
    """ Estrae il RT60 dall'impulse responce usando il metodo di Schroeder
    Ritorna:
    rt60: stima del RT60 in secondi
    """
    rt60 = pra.experimental.measure_rt60(impulse_responce, sr)
    return rt60



