import librosa
import numpy as np
import matplotlib.pyplot as plt

def envelope_extraction(y, sr, frame_length=2048, hop_length=512):
    """ Estrae l'inviluppo del segnale: divido il segnale in frame, calcolo l'energia media di ogni frame e ne calcolo la radice quadrata
    Ritorna:
    envelope: array con l'inviluppo del segnale in dB
    """
    #frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    energy_array = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    envelope = librosa.amplitude_to_db(energy_array, ref=np.max)

    return envelope

def plot_envelope(envelope, sr, hop_length=512):
    """Plotta l'inviluppo del segnale e lo salva nella cartella dei risultati """
    envelope1= envelope.flatten()
    time = librosa.frames_to_time(np.arange(len(envelope1)), sr=sr, hop_length=hop_length)
    plt.plot(time, envelope1)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Inviluppo (dB)")

def estimate_noise_floor(envelope, percentile=10):
    """
    Stima il livello del rumore di fondo dall'inviluppo.
    
    Usa il percentile basso dell'inviluppo: i frame più silenziosi
    rappresentano il rumore di fondo della registrazione.
    
    Parametri:
        envelope   : inviluppo in dB (output di envelope_extraction)
        percentile : percentile da usare (default 10 = prende i frame
                     più silenziosi ignorando i picchi)
    
    Ritorna:
        noise_floor_db : livello stimato del rumore di fondo in dB
    """
    env = envelope.flatten()
    noise_floor_db = np.percentile(env, percentile) #prende i frame più silenziosi ignorando il resto
    print(f"  ✓ Noise floor stimato: {noise_floor_db:.1f} dB")
    return noise_floor_db

def detect_pauses(envelope, sr, hop_length=512, min_pause_duration=0.1, noise_floor_margin_db=10, slope_threshold_db=-0.05):
    """Analizza la differenza di energia tra un frame e l'altro.
    Uso due margini: soglia inferiore data dal soglia di rumore e una soglia superiore per eliminare i picchi di energia
    Le pause utili saranno quelle dove la pendenza è negativa e costante per un certo tempo"""
    
    env = envelope.flatten()
    noise_floor = estimate_noise_floor(env)
    lower_thresh_db = noise_floor + noise_floor_margin_db  #considero che c'è una zona di incertezza, per questo uso un margine di sicurezza
    print(f"  ✓ Soglia silenzio adattiva: {lower_thresh_db:.1f} dB") 
     
    upper_threshold_db = np.percentile(env, 85) 

    valid_frames = (env < upper_threshold_db) & (env > lower_thresh_db) #considero solo i frame che sono sopra la soglia di rumore ma sotto la soglia dei picchi di energia

    """calcolo pendenza tra un frame e l'altro 
    considero i frame con pendenza negativa e costante"""

    env_valid = np.where(valid_frames, env, np.nan)
    slope = np.diff(env_valid, prepend=env_valid[0])

    valid_slope = slope < slope_threshold_db

    valid = valid_frames & valid_slope
    """Dizionario con pause rilevate: tempo di inizio della pausa, tempo di fine, 
    indice del frame iniziale e finale"""

    pauses =[]
    in_pause =False
    start =0
    end=0
    for i, value in enumerate(valid):
        if value and not in_pause:
            in_pause = True
            start = i
        
        elif not value and in_pause:
            in_pause = False
            end = i
            pause_duration = (end - start) * hop_length / sr
            if pause_duration >= min_pause_duration: #tengo pause abbastanza lunghe
                segmento_db = env[start:end]
                pauses.append({
                    "start_time": librosa.frames_to_time(start, sr=sr, hop_length=hop_length),
                    "end_time": librosa.frames_to_time(end, sr=sr, hop_length=hop_length),
                    "start_frame": start,
                    "end_frame": end,
                    "envelope_db":segmento_db
                })

    
    #caso in cui pausa arriva alla fine del segnale
    if in_pause:
        end = len(env)-1
        pause_duration= (end-start)*hop_length/sr
        if pause_duration>min_pause_duration:
            segmento_db = env[start:end]
            pauses.append({
                "start_time": librosa.frames_to_time(start, sr=sr, hop_length=hop_length),
                "end_time": librosa.frames_to_time(end, sr=sr, hop_length=hop_length),
                "start_frame": start,
                "end_frame": end,
                "envelope_db":segmento_db
            })
    return pauses