import matplotlib.pyplot as plt
import numpy as np
import os

def save_rt60_plot(valid_estimates, final_rt60_history, banda_nome, reference_rt60=1.16):
    """
    Salva i grafici in 'plot_results' seguendo lo stile Löllmann.
    """
    output_dir = "plot_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crea la figura con due subplot (Istogramma e Time-Evolution)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # 1. Istogramma delle stime grezze
    bins = np.arange(0.1, 4.0, 0.05)
    ax1.hist(valid_estimates, bins=bins, color='gray', alpha=0.5, label='Raw ML Estimates', density=True)
    
    # Linea della stima finale (il valore all'ultimo step)
    final_val = final_rt60_history[-1]
    ax1.axvline(final_val, color='blue', linewidth=2, label=f'Final Estimate: {final_val:.2f}s')
    ax1.axvline(reference_rt60, color='red', linestyle='--', label=f'Reference: {reference_rt60}s')
    
    ax1.set_title(f"Histogram of ML Estimates - Band {banda_nome} Hz")
    ax1.set_xlabel("RT60 [s]")
    ax1.set_ylabel("Probability Density")
    ax1.legend()

    # 2. Evoluzione temporale
    ax2.plot(final_rt60_history, color='blue', label='Smoothed RT60 (Löllmann)')
    ax2.axhline(reference_rt60, color='red', linestyle='--', label='Reference')
    
    ax2.set_title("Temporal Evolution of RT60 Estimate")
    ax2.set_xlabel("Number of valid estimates")
    ax2.set_ylabel("RT60 [s]")
    ax2.set_ylim(0, 3.0)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    # Salvataggio
    path = os.path.join(output_dir, f"plot_band_{banda_nome}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()