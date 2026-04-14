import librosa #libreria per gli audio 
import numpy as np #per lavorare con array del segnale
import soundfile as sf #

"""carica un file audio e lo ricampione alla frequenza target_sr
Parametri: 
- path: percorso del file audio da caricare
-targer_sr: frequenza di campionamento (default 44110 Hz)

Ritorna:
- y : segnale audio come array numpy (con valori tra -1 1)
- sr :frequenza di campionamento effettiva
"""
def load_audio(path, target_sr=44100):
    #si può aggiunegre validazione dell'input
    y, sr = librosa.load(path, sr=target_sr, mono= True)
    return y,sr


