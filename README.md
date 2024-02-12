
# AI Breast Cancer Classification
## Descrizione
Questo progetto si concentra sull'esplorazione di un dataset che contiene informazioni sulle caratteristiche delle cellule tumorali e le rispettive classificazioni come benigne o maligne. L'obiettivo principale del progetto è sviluppare un modello di apprendimento automatico in grado di classificare i tumori in base alle caratteristiche fornite nel dataset impiegando tecniche di machine learning.

Il dataset utilizzato è reperibile al seguente link: `[Breast Cancer Wisconsin Original Dataset]`.

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

## Librerie utilizzate

* **Numpy** importata come np come libreria ufficiale di Python con le quali si sono trattati tutti i dati e funzioni.

* **Pandas** importata come pd usata per la manipolazione e l'analisi dei dati.

* **Matplotlib.pyplot** importata come plt per fornire all'utenza una visualizzazione grafica.

Per l'installazione delle librerie è necessario digitare da terminale i seguenti comandi:

 * `pip install -r requirements.txt` per gestire le dipendenze.
 * Python legge il contenuto di requirements.txt e installa i pacchetti elencati con le versioni specificate.
 * `pip install pandas matplotlib` per Matplot.
 * `pip install numpy` per Numpy.
 * `pip install pandas` per Pandas.
 * `pip install seaborn` è una libreria di visualizzazione che fornisce un'interfaccia di grafi ad alto livello (basata sempre su Matplotlib).

  # 1. Data	Preprocessing
  Questo branch elabora il set di dati. Di seguito i suoi passaggi:
  
 - **Load the dataset:**
    Importiamo il dataset da un file chiamato `breast_cancer.csv`
        
- **Divide dataset in features and targets:**
    Estraiamo la colonna "Class" dal dataframe e la assegniamo alla variabile Y. Quindi assegniamo il resto del dataframe alla variabile X.

 - **Process missing values appropiately:**
    Dobbiamo tenere conto dei valori mancanti nel set di dati delle features. Per questo abbiamo utilizzato il metodo "blackfill".


 # 2. Model	Development
Di seguito i vari step che sono stai seguiti per questo branch:
## 1. Algoritmo utilizzato: K-Nearest Neighbors 
L'algoritmo K-nearest neighbor (KNN) è una tecnica di classificazione che opera valutando le caratteristiche di oggetti vicini a quello in esame. Il KNN riceve un set di dati di addestramento con etichette (classi) note. Per un nuovo punto di dati da classificare, calcola la sua distanza rispetto a tutti i punti di addestramento utilizzando una metrica di distanza (la più utilizzata è la distanza euclidea). In seguito identifica i k punti di addestramento più vicini al punto di test in base alla distanza calcolata. Per la classificazione, assegna all'istanza di test l'etichetta di classe più frequente tra i suoi k vicini più prossimi. 
### Due punti fondamentali:
* ### Distanza Euclidea:
Nel contesto di KNN, la distanza euclidea tra due punti è utilizzata per determinare la "vicinanza" tra di essi.
Nel contesto di KNN, quando si deve determinare la "vicinanza" di un punto di test a punti di addestramento, si calcolano le distanze euclidee tra il punto di test e tutti i punti di addestramento. Quindi, si selezionano i k punti di addestramento più vicini al punto di test in base alle distanze calcolate. L
In breve, la distanza euclidea in KNN è uno strumento fondamentale per determinare quanto i punti siano "vicini" o "simili" nel contesto delle loro caratteristiche, e viene utilizzata per predire l'etichetta o il valore di un punto di test basandosi sulle etichette o i valori dei suoi vicini più prossimi. 

* ### Parametro K:
  La selezione di un valore appropriato per k è una delle considerazioni chiave quando si utilizza KNN: non deve, infatti, essere nè troppo piccolo nè troppo grande per evitare che vada underfitting o overfitting, quindi va scelto un parametro k che, rispetto all'analisi che stiamo effettuando, sia il più performante possibile.

   ## 2. Funzione 'fit':
  La funzione `fit` viene utilizzata per addestrare il modello KNN sui dati di addestramento. Durante questo processo, il modello memorizza i dati di addestramento in 
  modo che possano essere utilizzati successivamente per fare previsioni.

  ## 3. Funzione 'predict':
  In un classificatore K-Nearest Neighbors (KNN) svolge il ruolo di fare previsioni su nuovi dati in base ai dati di addestramento memorizzati.
Grazie alla funzione `predict`, si calcolano le distanze tra i dati di test e i dati di addestramento utilizzando la funzione di distanza euclidea (self.euclidean_distance). Successivamente, si identificano i k vicini più prossimi e si determina l'etichetta predetta in base alle etichette di questi vicini.
Di seguito i passaggi:
* Creo una lista vuota (y_predictions) che verrà utilizzata per memorizzare le etichette predette per ciascun dato di test;
* Itera su ciascun dato di test presente nel DataFrame X_test;
* Si calcolano le distanze euclidee tra il dato di test corrente e tutti i dati di addestramento. Le distanze vengono memorizzate in una lista (distances) insieme alle rispettive etichette dei dati di addestramento;
* Si estraggono le etichette corrispondenti ai k vicini più prossimi;
* Si verifica se c'è una sola classe tra i k vicini più prossimi o se c'è un pareggio tra le occorrenze delle classi. In caso affermativo, si fa una scelta casuale tra le etichette coinvolte. Altrimenti, si seleziona l'etichetta più frequente;
* Infine, le previsioni vengono restituite sotto forma di un array NumPy;

  - `itemgetter` è una funzione nel modulo operator di Python che restituisce una funzione chiamabile che può essere utilizzata per ottenere un particolare elemento da un oggetto iterabile come una lista o una tupla. In particolare, la funzione itemgetter è utilizzata quando ordinamo le distanze tra i punti calcolati nella funzione euclidean_distance.
    
# 3. Model	Evaluation

# Valutazione del Modello implementato
## Descrizione
Questo modulo fornisce funzionalità per la valutazione del modello di classificazione del cancro al seno basato su k-Nearest Neighbors (KNN).
Si sono utilizzati due possibili tecniche di valutazione: Holdout e Cross-Validation.
Con Holdout si divide il dataset in 2 micro-set : addestramento e test su cui comparare i dati.
Con Cross-Validation il dataset si divide in k set che si comparano tra loro in maniera più dinamica fornendo più risultati.

## Holdout
In questo caso si fornisce un parametro k ovvero il numero dei vicini e si calcolano le metriche specifiche tramite le previsioni che si ricavano dal dataset. I dati sono randomicamente divisi in test e train. Nel codice si richiede all'utente di fornire un valore di test size affinchè l'Holdout venga effettuato in un determinato test_size. Questo garantisce la divisione del Dataset in train e test, l'optimum della divisione si ha circa con test_size = 0.2, 0.3
Con `random_state=None` otteniamo diversi set di training e test in diverse esecuzioni e il processo di mescolamento è fuori controllo.

## Leave one out Cross-Validation
In questa validazione, si tiene fuori un campione di test k e si addestra il modello nel train set restante. Effettua la validazione di ogni metrica per ogni singolo k, se l'utente fornisce un k > 1, il programma itera l'addestramento e di conseguenza il calcolo delle metriche k volte. Il risultato di questa validazione è la media delle metriche. 

- ## Metriche di Valutazione:
  Queste opzioni determinano come valutare le prestazioni del modello. Sono disponibili le seguenti metriche:

* `Accuracy Rate`: Percentuale di predizioni corrette rispetto al totale.
* `Error Rate`: Percentuale di predizioni errate rispetto al totale.
* `Sensitivity`: Capacità del modello di identificare correttamente i casi positivi.
* `Specificity`: Capacità del modello di identificare correttamente i casi negativi.
* `Geometric Mean`: Misura l'equilibrio tra Sensitivity e Specificity.
* `All the above` : Tutte le precedenti.

## File utilizzati per l' implementazione
* `fetching.py`: Modulo per il preprocessing e il caricamento dei dati.

* `knn.py`: Implementazione del classificatore KNN (KNNClassifier), funzioni di previsione e funzioni di utilità.

* `evaluation.py`: Script per valutare il modello KNN utilizzando Holdout e K-Fold Cross Validation.

* `breast_cancer.csv`: File contenente Dataset

- **Esecuzione del Programma**: Eseguire il file `main.py` specificando le opzioni di input come argomenti della linea di comando quando e come richieste.

- **Input richiesti**:
  * validation_type chiede all'utente che tipo di validazione tra Holdout e xx effettuare.
  * k = numero di esperimenti per xx, numero di vicini per Holdout.
  * test_size valore di test da considerare per Holdout.
  * metric_choice chiede all'utente quale metrica calcolare.

## Salvataggio e valutazione dei dati 
 Le metriche di ciascuna validazione vengono salvate in una cartella chiamata `output`, nella quale si crea un file csv chiamato `metrics.csv`.
 La creazione di questo file viene gestita dalla seguente funzione: 

        `def save_metrics(self, df):
        df.to_csv('output/metrics.csv', index=False)`
        
## Visualizzazione e interpretazione dei risultati

Se l'utente sceglie Holdout ed effettua una divsione del train e test sets in maniera poco proporzionata è possibile avere dei risultati falsati.

Con la LOOCV è possibile superare questo "limite" e gestire in maniera dinamica e rapida la valutazione.

L'utente può visualizzare e di conseguenza può comprendere la valutazione complessiva dell'analisi richiesta grazie alla creazione di file di plots nei quali sono presenti dei grafici riportanti le metriche calcolate al fine di rappresentare le rispettive distribuzione nei vari esperimenti. 



  
