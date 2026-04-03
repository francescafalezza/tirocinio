#il T60 si misura in diverse bande di frenqueza (ISO 3382) perchè varia con la frequenza

import numpy as np
from scipy.signal import butter, sosfilt

OCTAVE_CENTER_FREQUENCIES = [125, 250, 500, 1000, 2000, 4000]

def octave_band_edges (center_freq):
    """Calcola i limiti superiori e inferiori per una banda di ottava
    limite inferiore è center_freq/sqrt2, limite superiore è center_freq*sqrt2
    ritorna:
    low : frequenza di taglio superiore

    high: frequenza di taglio inferiore
    """
    low = center_freq/np.sqrt(2)
    high = center_freq*np.sqrt(s)
    return low, high

def octave_filter(center_freq, sr, order=4):
    """ Crea filtro passa-banda Butterworth per una banda di ottava
    Ritorna: coeficienti del filtro in formato second-order sections
    """

    nyq = sr/2.0 #frequenza di Nyquisit

    low, high = octave_band_edges(center_freq)
    
    #normalizzo le frequenze (scipy vuole valori tra 0 e 1)
    low_norm = low/nyq
    high_norm = high/nyq

    #controllo che le frequenze siano nel range valido
    if low_norm <=0 or high_norm >=1:
        raise ValueError(f"Frequenzedi taglio fuori dal range valido")
    
    #progetto filtro passa banda Butterworth
    sos = butter(order, [low_norm, high_norm], btype = 'band', output= 'sos')
    return sos

def applay_filter(y, center_freq, sr, order=4):
    """Applica i filtro a banda di ottava al segnale
    Ritorna:
    y_filtered: segnale filtrato nella banda"""

    sos = octave_filter(center_freq, sr, order)
    y_filtered = sosfilt(sos, y) 
    return y_filtered

def filter_all_bands(y, sr, center_freq=None, order=4):
    """Filtra il segnale per tutte le bande di ottava 
    Ritorna: dizionario con segnale filtratro per ogni banda (frequenza centrale:segnale_filtrato)"""


    if center_freq is None:
        center_freq = OCTAVE_CENTER_FREQUENCIES

    bands = {}

    for freq in center_freq:
        try:
            y_filtered = applay_filter(y, freq, sr, order)
            bands[freq] = y_filtered
            prinf(f"Banda {freq} Hz filtrata")
        except ValueError as e:
            print(f"Errore per la frequenza {freq}: {e}")

    return bands
