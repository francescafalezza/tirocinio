flowchart TD
    %% Inizio
    Start([Inizio]) --> Input[Input Utente: Carica RIR e Registrazione]
    
    %% Pre-processing
    Input --> Filtering[Filtraggio della registrazione per bande di frequenza]
    Filtering --> Choice{Scelta Algoritmo}

    %% Altri Algoritmi
    Choice -- Altro --> Other[Esecuzione altro algoritmo...]
    
    %% Ramo Spectral Flatness
    Choice -- Spectral Flatness --> SF_Down[Downsampling per bande]
    
    subgraph Spectral_Flatness_Logic [Logica Spectral Flatness]
        SF_Down --> SF_Pauses[Detect Pauses]
        SF_Pauses --> SF_Pre[Lollmann Preselection]
        SF_Pre --> SF_Weight[Selection Weighted]
        SF_Weight --> SF_Cand[Estimate All Candidates]
        SF_Cand --> SF_Final[Compute Final RT60]
    end

    %% Conclusione
    SF_Final --> Output[Mostra/Salva Risultati RT60]
    Other --> Output
    Output --> End([Fine])

    %% Stile
    style Spectral_Flatness_Logic fill:#f9f,stroke:#333,stroke-width:2px
    style Choice fill:#fff4dd,stroke:#d4a017