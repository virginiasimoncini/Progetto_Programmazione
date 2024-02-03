# Valutazione del Modello implementato
## Descrizione
Questo modulo fornisce funzionalità per la valutazione del modello di classificazione del cancro al seno basato su k-Nearest Neighbors (KNN).
Si sono utilizzati due possibili tecniche di valutazione: Holdout e Cross-Validation.
Con Holdout si divide il dataset in 2 micro-set : addestramento e test su cui comparare i dati.
Con Cross-Validation il dataset si divide in k set che si comparano tra loro in maniera più dinamica fornendo più risultati.

Si importano le logiche implementate negli altri branch rispettivamente il caricamento del dataset fornito e la logica per la gestione e organizzazione del processo.

## Holdout
In questo caso si fornisce un parametro k ovvero il numero dei vicini e si calcolano le metriche specifiche tramite le previsioni che si ricavano dal dataset.
Si utilizza random_state=42 per fornire all'utente una combinazione che effettua 42 possibili comparazioni in maniera tale da seguire lo sviluppo di ognuna e confrontarla quando si cambiano i test e train.
A partire da queste calcolo l'efficienza della mia analisi e conservo le metriche in un file csv.

## Cross-Validation
Nel caso del Cross-Validation si analizza l'accuratezza di ogni corrispondenza e si fa una media che fornisce per l'appunto l'andamento generale dei test.
Con questo indices = np.random.permutation(features.index) si va ad effettuare un mix casuale di test che si riproduce in maniera casuale ma più robusta.
OLa media delle accuretezze viene riportata in un csv.

