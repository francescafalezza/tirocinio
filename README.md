
Analisi comparativa di due algoritmi diversi per la stima cieca del tempo di riverbero.


Per il corretto funzionamento del software è necessario possedere una registrazione qualsiasi di una sala e la RIR corrispondente. 

L'implementato si basa sull'articolo "An improved algorithm for blind reverberation time estimation" di Lollmann et all.

L'algoritmo di Lollmann parte da una ipotesi statistica: il suono che decade si comporta come un processo casuale dove l'energia diminuisce in modo esponenziale nel tempo. Il segnale viene visto come una sequenza di variabili casuali con distribuzione normale, la larghezza della distribuzione di resistrenge con il tempo, seguendo il tasso di decadimento. 

Il metodo della funzione Max-likelyhood si rifà al metodo di Ratman. la funzione viene implementata usando la classe BlindRT_60 della repository di github  https://github.com/nuniz/blind_rt60
Viene aggiunto: un processo di downsampling del segnale, una preselezione dei possibili frame di decadimento e due istogrammi che raccolgono le stime del rt60 per rilevare variazioni rapide.  Il secondo istogramma viene utilizzato nel seguente modo:se le stime del secondo istogramma variano rispetto a quelle del primo per un periodo maggiore di una soglia predefinita, vengono usate le stime del secondo istogramma, il primo viene riempito con queste ultime e viene utilizzato un fattore si smoothing basso per permettere alle stime di seguire la variazione. Altrimenti viene usato un fattore smoothing maggiore, per mantenere le stime stabili. 

In plot_results vengono salvati i grafici delle stime del rt60 per ogni banda di frequenza. 


Differenze di implementazione: 
1. Nell'articolo l'algoritmo viene testato su segnali di parlato. Il software è pensato per funzionare anche con segnali musicali. 
2. Nell'articolo viene usato un fattore di framing M=128. Nell'algoritmo presentato con un fattore così piccolo non sono visibili risultati. Bisogna usare M>512  questo perchè il modello ML richiede un numero sufficiente di campioni del suono per stimare il tasso di decadimento e migliorare la stima del rt60 poichè una finestra più ampia riduce l'influenza del rumore. Per RT bassi (come quelo presentato nell'articolo) è sufficiente un M=128.

I risultati dell'algoritmo "lollman" presentano un errore medio del 15%.
