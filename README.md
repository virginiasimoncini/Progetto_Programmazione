# AI Breast Cancer Classification
## Descrizione
Questo progetto si concentra sull'esplorazione di un dataset che contiene informazioni sulle caratteristiche delle cellule tumorali e le rispettive classificazioni come benigne o maligne. L'obiettivo principale del progetto è sviluppare un modello di apprendimento automatico in grado di classificare i tumori in base alle caratteristiche fornite nel dataset impiegando tecniche di machine learning.

Il dataset utilizzato è reperibile al seguente link: [Breast Cancer Wisconsin Original Dataset].

È importante sottolineare che l'obiettivo del progetto non è necessariamente quello di ottenere un modello ad alte prestazioni, ma piuttosto di creare una pipeline di machine learning che possa essere personalizzata dall'utente per addestrare e valutare un modello di classificazione dei tumori sulla base delle caratteristiche del dataset fornito. Questo approccio mira a fornire una flessibilità maggiore all'utente finale nella definizione delle caratteristiche rilevanti per la classificazione.

#Dataset breast_cancer.csv
Il dataset breast_cancer.csv utilizzato in questo progetto è fondamentale per l'analisi e la classificazione dei tumori. 
Di seguito una panoramica dettagliata del dataset:

**Numero di campioni:**  699 campioni
**9	variabili	indipendenti (features):** 

* Clump	Thickness:	1	– 10
* Uniformity	of	Cell	Size:	1	– 10
* Uniformity	of	Cell	Shape:	1	– 10
* Marginal	Adhesion:	1	– 10
* Single	Epithelial	Cell	Size:	1	– 10
* Bare	Nuclei:	1	– 10
* Bland	Chromatin:	1	– 10
* Normal	Nucleoli:	1	– 10
* Mitoses:	1	– 10

**Classe**: Classificazione del tumore (2 per benigno, 4 per maligno).

## Dipendenze necessarie per il codice
In ogni parte del codice si sono utilizzate le principali librerie che Python offre:

**Numpy** importato as np tramite la quale si sono trattati tutti i dati e funzioni.

**Pandas** importato as pd per la manipolazione e la gestione dei dati.

Le dipendenze sopra riportate si possono installare tramite  `pip install -r requirements.txt` direttamente dal terminale, con questo comando sarà possibile gestire le principali librerie.

## Analisi del codice
Abbiamo suddiviso il nostro codice in tre differenti passaggi: 

1. **Data	Preprocessing:**
Questo branch elabora il dataset. Di seguito i passaggi:
    - **Load the dataset:**
       Importiamo il set di dati dal file chiamato `breast_cancer.csv` già compilato e gestito.
    - **Divide dataset in features and targets:**
Estraiamo la colonna "Classe" dal dataframe e la assegniamo alla variabile Y. Quindi assegniamo il resto del dataframe alla variabile X. Con questa operazione abbiamo potuto gestire le variabili indipendenti e quelle dipendenti.
    - **Process missing values appropiately:**
Dobbiamo tenere conto dei valori mancanti nel set del dataset delle features. Per questo abbiamo utilizzato il metodo "blackfill".

2. **Model	Development**

3. **Model	Evaluation:**

