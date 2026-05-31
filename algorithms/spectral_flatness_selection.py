
import numpy as np
from algorithms import lollmann
from blind_rt60 import BlindRT60
from collections import deque
import math

def calcola_sf(subframe):
    """Calcola la Spectral Flatness manualmente su un array 1D."""
    # FFT e power spectrum
    psd = np.abs(np.fft.rfft(subframe)) ** 2
    psd = psd + 1e-10  # evita log(0)
    
    # media geometrica tramite log per stabilità numerica
    log_mean = np.exp(np.mean(np.log(psd)))
    lin_mean = np.mean(psd)
    
    return log_mean / lin_mean


def spectral_flatness_selection_weighted(decay_candidates, L=4, slope_threshold=None):
    """Funzione che calcola la spectral flatness su ogni subframe e ne guarda la crescita. 
    Da usare nella preselezione fatta nell'algortimo di Lollman. 
    Parametri: 
    - decay candidates già pre-selezionati da analizzare
    -L lunghezza subframe
    -slope_treshold pendenza minima della regressione lineare per accettare il frame
    
    Ritorna: frame con SF che cresce nel tempo in modo monotono"""
    
    #SF deve crescere in modo monotono per ogni subframe da l=0 a l=L-1
    
    l_start=math.ceil(L/2) #applico sf sugli ultimi L/2 subframe, dove è più probabile ci sia solo coda riverberante
    final_candidates=[]
    slope_and_frames=[]
    
    for segment in decay_candidates:
        segment = np.array(segment)
        N=len(segment)
        P=N/L
        SF_subframe=[]
        
        for l in range(1, L-1):
            
            subframe = segment[int(l*P):int((l+1)*P)]
            SF_subframe = [f for f in SF_subframe if np.isfinite(f)]
            
            
            if len(subframe)<128:
                continue
            SF_subframe.append(calcola_sf(subframe))
        
        if len(SF_subframe)<2: #servono alemno due punti per la pendenza
                continue
        
        #regressione lineare sulla sequenza di sf
        x =np.arange(len(SF_subframe))
        slope, _= np.polyfit(x, SF_subframe, 1)
        slope_and_frames.append((slope,segment))
        #se la pendenza è negativa, SF diminuisce, peso 0
        #se è positiva, il peso è la pendenza
       
        if len(slope_and_frames)==0:
            return [],[]
        
     
    slopes = [s for s, _ in slope_and_frames]
    pendenze_positive = [s for s in slopes if s > 0]
    
    if slope_threshold is None and len(pendenze_positive) > 0:
        slope_threshold = np.percentile(pendenze_positive, 25)
    elif slope_threshold is None:
        slope_threshold = 0.0
    
    weights = []
    for slope, segment in slope_and_frames:
        if slope > slope_threshold:
            final_candidates.append(segment)
            # peso = pendenza normalizzata tra 0 e 1
            weights.append(slope)
    
    # normalizza i pesi tra 0 e 1
    if len(weights) > 0:
        w_min = min(weights)
        w_max = max(weights)
        if w_max > w_min:
            weights = [(w - w_min) / (w_max - w_min) for w in weights]
        else:
            weights = [1.0] * len(weights)
    
    print(f"Candidati accettati: {len(final_candidates)}/{len(slope_and_frames)}")
    
    # ritorna sia i candidati che i loro pesi
    return final_candidates, weights

    
def estimate_all_candidates_SF(candidates, weights, sr_downsampled):
    
    if not candidates:
        print("NESSUN CANDIDATO")
        return [],[] 
    
    rt60_estimates = []
    rt60_weights = []
    
    
    expected_len=len(candidates[0])
        
    estimator = BlindRT60(
        fs=sr_downsampled,
        framelen=expected_len / sr_downsampled, # N dinamico
        max_itr=500,
        max_err=1e-1,
        bisected_itr=8,
        a_init=0.99
        )
    

    for segment, weight in zip(candidates, weights):
        
        segment = np.array(segment)
        
        if len(segment) != expected_len:
            continue
        
        rt60 = lollmann.ML_estimate_one_candidate(segment, sr_downsampled, estimator)
        
        if rt60 is not None:
            if rt60>0.0 and rt60<3.0:
                rt60_estimates.append(rt60)
                rt60_weights.append(weight)
    
    return rt60_estimates, rt60_weights
    
    """
    for segment, weight in zip(candidates, weights):
        segment = np.array(segment)
        
        if len(segment)!= expeceted_len:
            continue
        
        rt60 = lollmann.ML_estimate_one_candidate(segment, sr_downsampled, estimator)
        
        if rt60 is not None:
            rt60_estimates.append(rt60)
            rt60_weights.append(weight)
    
    return rt60_estimates, rt60_weights """




def update_histograms_SF(
    rt60_estimate: float,
    sf_weight: float,          # aggiungi questo parametro
    buffer_slow: deque,
    buffer_fast: deque,
    bin_edges: np.ndarray,
) -> tuple:
    
    # invece di appendere una volta sola, appendi
    # un numero di volte proporzionale al peso
    # es. peso 1.0 → appendi 10 volte, peso 0.1 → appendi 1 volta
    n_copies = max(1, round(sf_weight * 50))
    
    for _ in range(n_copies):
        buffer_slow.append(rt60_estimate)
        buffer_fast.append(rt60_estimate)

    # resto invariato
    enough_data = len(buffer_fast) >= max(3, buffer_fast.maxlen // 2)
    
    if not enough_data:
        return rt60_estimate, rt60_estimate, False

    counts_slow, _ = np.histogram(list(buffer_slow), bins=bin_edges)
    counts_fast, _ = np.histogram(list(buffer_fast), bins=bin_edges)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_slow = bin_centers[np.argmax(counts_slow)]
    peak_fast = bin_centers[np.argmax(counts_fast)]

    return peak_slow, peak_fast, True

def compute_final_rt60_SF(
    valid_estimates: list,
    sf_weights: list,          # parametro aggiunto
    kf: int = 400,
    ks: int = 20,
    eps_t: float = 0.2,
    q_threshold: int = 100,
    bin_edges: np.ndarray = None,
) -> float | None:
    
    if len(valid_estimates) < ks:
        # anche qui usa i pesi per la mediana pesata
        weighted_median = np.average(valid_estimates, weights=sf_weights)
        return [float(weighted_median)]

    if bin_edges is None:
        bin_edges = np.arange(0.1, 4.15, 0.15)

    buffer_slow = deque(maxlen=kf)
    buffer_fast = deque(maxlen=ks)
    current_rt60 = None
    consecutive_diff = 0
    history = []

    # itera su stime E pesi insieme
    for rt60, weight in zip(valid_estimates, sf_weights):

        peak_slow, peak_fast, enough_data = update_histograms_SF(
            rt60, weight,          # passa il peso
            buffer_slow, buffer_fast, bin_edges
        )

        if not enough_data:
            current_rt60 = rt60
            history.append(current_rt60)
            continue

        diff = abs(peak_slow - peak_fast)

        if diff > eps_t:
            consecutive_diff += 1
        else:
            consecutive_diff = 0

        if consecutive_diff >= q_threshold:
            peak_to_use = peak_fast
            beta = 0.99
            buffer_slow.clear()
            buffer_slow.extend(list(buffer_fast))
            consecutive_diff = 0
        else:
            peak_to_use = peak_slow
            beta = 1 - (1 / min(kf, len(valid_estimates)))

        if current_rt60 is None:
            current_rt60 = peak_to_use
        else:
            current_rt60 = beta * current_rt60 + (1 - beta) * peak_to_use

        history.append(current_rt60)

    print(f"Peso minimo SF: {min(sf_weights):.4f}")
    print(f"Peso massimo SF: {max(sf_weights):.4f}")
    print(f"Peso medio SF: {np.mean(sf_weights):.4f}")
    print(f"Stima minima RT: {min(history):.3f} s")
    print(f"Stima massima RT: {max(history):.3f} s")
    return history

