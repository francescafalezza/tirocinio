#chiede il path della registrazione e della rir all'utente e chiama la funzione load del loader x 2
#chiama le funzioni di preprocessing sui file caricati
#manda la rir a referepreprocessing import loader
import algorithms as alg
from preprocessing import loader
from preprocessing import band_filter
from preprocessing import pause_detector 
from algorithms import lollmann 
from algorithms import reference

def main():
    path_rir=input("Inserire path della RIR: ")
    path_recording=input("Inserire path della registrazione: ")
    
    rir, sr_rir = loader.load_audio(path_rir)
    recording, sr_recording = loader.load_audio(path_recording)
    
    #RT di riferimento
    rt_reference = reference.extract_ground_truth_rt60(rir, sr_rir)
    print(f"RT60 di riferimento: {rt_reference:.2f} secondi")
    
    #Metodo Lollman 
    
    #Step 2: filtraggio per bande
    bands = band_filter.filter_all_bands(recording, sr_recording)
    
    rt60_per_band= {}
    
    for freq, recording_filtred in bands.items():
        print(f"\nProcesso banda {freq} Hz")
        signal_band_downsampled, sr_downsampled = lollmann.downsampling(recording_filtred, sr_recording, R=5)
    
        envelope = pause_detector.envelope_extraction(signal_band_downsampled, sr_downsampled)
        
        pauses = pause_detector.detect_pauses (envelope, sr_downsampled)
        print(f"Pause trovate: {len(pauses)}")
        
        if len(pauses)==0:
            print(f"\nNessuna pausa trovata per banda {freq} Hz, salto stima RT60")
            continue
        
        #candidati di decay
        decay_candidates= lollmann.extract_decay_candidates(signal_band_downsampled, sr_downsampled, pauses)
        print(f"Candidati estratti : {len(decay_candidates)}")
        if len(decay_candidates)==0:
            print(f"\nNessun candidato per banda {freq} Hz.")
            
            
        #ML esimation 
        raw_estimation= lollmann.estimate_all_candidates(decay_candidates, sr_downsampled)
        valid = [rt for rt in raw_estimation if rt is not None]
        print(f"  Stime valide: {len(valid)} su {len(raw_estimation)}")
        print (f"Stime RT60 per banda {freq} Hz: {raw_estimation}")
        
        if len(valid) == 0:
            print(f"  Nessuna stima valida per banda {freq} Hz")
            continue

        rt60_final = lollmann.compute_final_rt60(valid)
        rt60_per_band[freq] = rt60_final
        print(f"  RT60 stimato banda {freq} Hz: {rt60_final:.2f} s")
        
if __name__ == "__main__":
    main()