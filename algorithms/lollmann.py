import scipy.signal 
import numpy as np
from blind_rt60 import BlindRT60
from collections import deque
import librosa


class UpdateMethod:
    NEWTON = "newton"
    BISECTED = "bisected"


def downsampling (signal,sr, R):
    """Decimazione del segnale, riduce frequenza di campionamento 
    (per rideurre carico computazionale rispetto al metodo Ratman)"""
    signal_downsampled=scipy.signal.decimate(signal,R)
    sr_downsampled=sr//R
    return signal_downsampled, sr_downsampled

"""
def extract_decay_candidates(signal, sr, pauses, hop_length=512, 
                              decay_window_s=0.3):
    
    Per ogni pausa rilevata, estrae la finestra di segnale 
    immediatamente precedente come candidato al sound decay.
    
    Il decay avviene negli ultimi ~200-300ms prima del silenzio.
    
    decay_window_samples = int(decay_window_s * sr)
    candidates = []
    
    

    for pause in pauses:
        # Il decay finisce dove inizia la pausa
        decay_end = int(pause["end_time"] * sr)
        decay_start = int(pause["start_time"] * sr)
        
        segment = signal[decay_start:min(decay_end, decay_start+decay_window_samples)]
        
        # Verifica minima: il segmento deve avere abbastanza campioni
        if len(segment) > int(0.05 * sr):  # almeno 50ms
            candidates.append({
                "signal": segment,
                "start_sample": decay_start,
                "end_sample": decay_end,
                "pause_ref": pause,
                "envelope_db":pause["envelope_db"]
            })
    
    return candidates
"""
def ML_estimate_one_candidate(segment: np.ndarray, sr_downsampled: int, estimator: BlindRT60) -> float |None:
    """Stima il RT60 per un singolo segmento di decay tramite ML
     Parametri:
        segment: array 1D del segnale audio del decay del pause detector
        sr_downsampled: sample rate del segnale downsampling

    ritorna:
        rt60: stima in secondi oppure None è fuori dai range
    """
    

    #step() vuole x_frames di shape (batch, framelen)=(1,N)
    #uso lunghezza del segmento come framelen, così self.n in likelohood_derivate ha la dimensione giusta

    N =len(segment)
    x_frame_2d= segment.reshape(1,N)
    
   

    estimator.init_states(batch=1)

    #lopp di convergenza: si ferma quando dl_da <max_err
    converged = False

    for itr in range(estimator.max_itr):
    
        method=(
            UpdateMethod.BISECTED
            if itr<estimator.bisected_itr
            else UpdateMethod.NEWTON
        )

        dl_da= estimator.step(x_frame_2d, method=method)

        if abs(dl_da)<= estimator.max_err:
            converged=True
            break
        
        #se non converge scartiamo la stima
    if not converged:
        return None
        
    #leggo self.a e lo converto in RT60. Tau costante di tempo di decadimento
    a= estimator.a.item()
    
    if a<=0 or a>=1:
        return None
    tau = -1.0/(np.log(a)*sr_downsampled)

    rt60= 6.908*tau


    return rt60



def estimate_all_candidates(decay_candidates, sr_downsampled:int):

    """Applica estimate_one_candidate a tutti i candidati
    Paramentri
    decay_candidates: output di estract_decay_candidates
    sr = sanmple rate originale
    
    Ritorna
    lista di float"""
    rt60_estimates= []
     #inizializzaione di BlindRT60
    estimator= BlindRT60(
        fs=sr_downsampled,
        framelen=1024/sr_downsampled,
        max_itr=500,
        max_err=1e-1,
        bisected_itr=8,  #prime 8 iterazioni con bisezione
        a_init =0.99,     #poi Newton
    )
    for i, candidate in enumerate(decay_candidates):
        segment = candidate
        rt60=ML_estimate_one_candidate(segment, sr_downsampled, estimator)
        if rt60 is not None:
            if rt60>0 and rt60<3.0:
                rt60_estimates.append(rt60)
        
    return rt60_estimates




def update_histograms(
    rt60_estimate: float,
    buffer_slow: deque,
    buffer_fast: deque,
    bin_edges: np.ndarray,
) -> tuple:
    """
    Aggiorna i due buffer con la nuova stima e calcola i picchi.
    
    Parametri:
        rt60_estimate : nuova stima RT60 dalla ML
        buffer_slow   : deque con maxlen=Kf, istogramma lento
        buffer_fast   : deque con maxlen=Ks, istogramma veloce, reattivo ai cambinamenti di RT
        bin_edges     : array con i bordi dei bin, uguale per entrambi (entrambi gli istogrammi usano gli stessi intervalli)
    
    Ritorna:
        peak_slow : RT60 corrispondente al picco dell'istogramma lento
        peak_fast : RT60 corrispondente al picco dell'istogramma veloce
        enough_data: True se il buffer veloce è pieno abbastanza
                     per avere un picco affidabile
    """
    # Aggiorno entrambi i buffer — deque con maxlen
    
    buffer_slow.append(rt60_estimate)
    buffer_fast.append(rt60_estimate)

    # Serve almeno metà del buffer veloce per avere un picco affidabile
    
    enough_data = len(buffer_fast) >= max(3, buffer_fast.maxlen // 2)

    if not enough_data:
        # Non abbastanza dati — ritorna la stima grezza come picco provvisorio
        return rt60_estimate, rt60_estimate, False

    # Costruisco i due istogrammi con gli stessi bin_edges
    counts_slow, _ = np.histogram(list(buffer_slow), bins=bin_edges)
    counts_fast, _ = np.histogram(list(buffer_fast), bins=bin_edges)

    # Il picco è il centro del bin con più occorrenze
    # bin_edges ha N+1 elementi, i centri sono la media di bordi contigui
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    peak_slow = bin_centers[np.argmax(counts_slow)]
    peak_fast = bin_centers[np.argmax(counts_fast)]

    return peak_slow, peak_fast, True


def compute_final_rt60(
    valid_estimates: list,
    kf: int = 400,
    ks: int = 20,
    eps_t: float = 0.2,
    q_threshold: int = 100,
    bin_edges: np.ndarray = None,
) -> float | None:
    """
    Doppio istogramma con smoothing adattivo — Löllmann 
    
    Parametri:
        valid_estimates : lista di float, stime ML valide (no None)
        kf              : dimensione buffer lento  (default 400) (valori usati da Löllmann)
        ks              : dimensione buffer veloce (default 20)
        eps_t           : soglia differenza picchi per rilevare cambio RT
        q_threshold     : frame consecutivi di differenza per confermare cambio
        bin_edges       : bordi dei bin, default 0.1s-4.0s step 0.05s
    
    Ritorna:
        rt60_finale : float con stima finale, None se non ci sono abbastanza dati
    """
   
    # Con meno di ks stime non ha senso usare il doppio istogramma —
    # ritorna direttamente la mediana come stima robusta
    if len(valid_estimates) < ks:
        return float(np.median(valid_estimates))

    if bin_edges is None:
        bin_edges = np.arange(0.1, 4.05, 0.05)

    # Inizializza i due buffer circolari
    buffer_slow = deque(maxlen=kf)
    buffer_fast = deque(maxlen=ks)

    # Stato dello smoothing
    current_rt60 = None
    consecutive_diff = 0        # contatore frame con differenza > eps_t
    history=[]
    for rt60 in valid_estimates:

        peak_slow, peak_fast, enough_data = update_histograms(
            rt60, buffer_slow, buffer_fast, bin_edges
        )

        # Finché non ci sono abbastanza dati nei buffer
        # usa la stima grezza e continua
        if not enough_data:
            current_rt60 = rt60
            history.append(current_rt60)
            continue

        # --- Logica di switch tra istogramma lento e veloce ---
        diff = abs(peak_slow - peak_fast)

        if diff > eps_t:
            consecutive_diff += 1
        else:
            consecutive_diff = 0

        if consecutive_diff >= q_threshold:
            # Il RT sta cambiando — switch all'istogramma veloce
            peak_to_use = peak_fast
            beta = 0.99                     # smoothing basso, insegue velocemente il nuovo valore

            # Resetta il buffer lento con i valori del veloce
            
            buffer_slow.clear()
            buffer_slow.extend(list(buffer_fast))
            consecutive_diff = 0

        else:
            # Condizioni stabili — usa l'istogramma lento
            peak_to_use = peak_slow
            beta = 1-(1/min(kf, len(valid_estimates)))                 # smoothing forte, bassa varianza

        # --- Smoothing ricorsivo Eq. 15 di Löllmann ---
        # T60(t) = beta * T60(t-1) + (1-beta) * T60_picco(t)
        if current_rt60 is None:
            current_rt60 = peak_to_use
        else:
            current_rt60 = beta * current_rt60 + (1 - beta) * peak_to_use

        history.append(current_rt60)
        
    return history 


def pre_selection(signal_downsampled):

    """ 
    Step 1: framing : segnale diviso in frame (M=128), segnale processato per frame di lunghezza M spostati di M_a
    Step 2: framing, ogni frame è diviso in L=7 subframe, ogni subframe è lungo P= 18 campioni 
    Per ciascun sotto-frame si calcola:
        -energia 
        -valore massimo 
        -valore minimo
        
    CONFRONTO CON SUB FRAME SUCC: un possibile decadimento è dichiarato quando per l=0, ..L-2 sono soddisfratte tre condizioni 
    1. energia del sotto frame attuale è maggiore del successivo
    2. valore del max deve diminuire (garantisce che le creste più evidenti smorzino con il tempo)
    3. valore del minimo deve calare (il valore minimo deve aumentare verso zero, cioè il valore più negativo
    del frame corrente deve essere "meno negativo" del successivo)
    
    Solo se le tre condizioni sono soddifatte per una sequenza consecutiva di sottoframe lmin=3
    
    Se una delle condizioni è violata, si contralla se l ha raggiunto un valore minimo, altrimenti si scarta quel frame e si passa al successivo 
    
    il vettore di pesi è scelto fra 0.8-1 
    """
    #framing e subframing
    M =1024
    L =28
    M_a=25 
    P=18
    w_var=0.8
    w_max=0.8
    w_min=0.8
    count=0
    l_min=3
    
    sub_frames=[]
    decay_candidate=[]
    
    for i in range(0,len(signal_downsampled)-M, M_a):
        current_frame=signal_downsampled[i:i+M] #prende il frame intero
        count=0
        
        for l in range(L-1):    
        #verifico tre condizioni per ogni subframe
            y_curr = current_frame[l*P : (l+1)*P]
            y_next = current_frame[(l+1)*P : (l+2)*P]
            energy_possible_candidate=np.sum(np.square(y_curr)) 
            max_energy_value=np.max(np.abs(y_curr))
            min_energy_value=np.min(np.abs(y_curr))
        
            energy_next_subframe=np.sum(np.square(y_next))
            max_energy_next_subframe=np.max(np.abs(y_next))
            min_energy_next_subframe=np.min(np.abs(y_next)) 
    
            cond_1=energy_possible_candidate> (w_var*energy_next_subframe)
            cond_2=max_energy_value > (w_max*max_energy_next_subframe)
            cond_3=min_energy_value < (w_min*min_energy_next_subframe)
        
        
            if cond_1 and cond_2 and cond_3:
                count=count+1
            
            else:
                if count>=l_min:
                    break
                
                count=0
                
        if(count>=l_min):
            decay_candidate.append(current_frame)
        
    type(decay_candidate)
    print(f"Candidati di decay pre-selezionati: {len(decay_candidate)}")
        
    return decay_candidate