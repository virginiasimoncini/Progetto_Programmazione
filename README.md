# Model	Development
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
  La funzione fit viene utilizzata per addestrare il modello KNN sui dati di addestramento. Durante questo processo, il modello memorizza i dati di addestramento in modo 
  che possano essere utilizzati successivamente per fare previsioni.

  ## 3. Funzione 'predict':
  In un classificatore K-Nearest Neighbors (KNN) svolge il ruolo di fare previsioni su nuovi dati in base ai dati di addestramento memorizzati.
Grazie alla funzione predict, si calcolano le distanze tra i dati di test e i dati di addestramento utilizzando la funzione di distanza euclidea (self.euclidean_distance). Successivamente, si identificano i k vicini più prossimi e si determina l'etichetta predetta in base alle etichette di questi vicini.
Di seguito i passaggi:
* Creo una lista vuota (y_predictions) che verrà utilizzata per memorizzare le etichette predette per ciascun dato di test;
* Itera su ciascun dato di test presente nel DataFrame X_test;
* Si calcolano le distanze euclidee tra il dato di test corrente e tutti i dati di addestramento. Le distanze vengono memorizzate in una lista (distances) insieme alle rispettive etichette dei dati di addestramento;
* Si estraggono le etichette corrispondenti ai k vicini più prossimi;
* Si verifica se c'è una sola classe tra i k vicini più prossimi o se c'è un pareggio tra le occorrenze delle classi. In caso affermativo, si fa una scelta casuale tra le etichette coinvolte. Altrimenti, si seleziona l'etichetta più frequente;
* Infine, le previsioni vengono restituite sotto forma di un array NumPy;

  
=========
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

## Analisi del codice
Abbiamo suddiviso il nostro codice in tre differenti passaggi: 

 # 1. Data	Preprocessing
This branch process the dataset.Its steps are:
    - **Load the dataset:**
        We import the dataset from a file called `breast_cancer.csv`
    - **Divide dataset in features and targets:**
We pop the "Class" column from the dataframe and assign it to the Y variable. Then we assign the rest of the dataframe to the X variable.
    - **Process missing values appropiately:**
We must take into account the missing values in the features dataset. For this we have used the "blackfill" method.

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
  La funzione fit viene utilizzata per addestrare il modello KNN sui dati di addestramento. Durante questo processo, il modello memorizza i dati di addestramento in modo 
  che possano essere utilizzati successivamente per fare previsioni.

  ## 3. Funzione 'predict':
  In un classificatore K-Nearest Neighbors (KNN) svolge il ruolo di fare previsioni su nuovi dati in base ai dati di addestramento memorizzati.
Grazie alla funzione predict, si calcolano le distanze tra i dati di test e i dati di addestramento utilizzando la funzione di distanza euclidea (self.euclidean_distance). Successivamente, si identificano i k vicini più prossimi e si determina l'etichetta predetta in base alle etichette di questi vicini.
Di seguito i passaggi:
* Creo una lista vuota (y_predictions) che verrà utilizzata per memorizzare le etichette predette per ciascun dato di test;
* Itera su ciascun dato di test presente nel DataFrame X_test;
* Si calcolano le distanze euclidee tra il dato di test corrente e tutti i dati di addestramento. Le distanze vengono memorizzate in una lista (distances) insieme alle rispettive etichette dei dati di addestramento;
* Si estraggono le etichette corrispondenti ai k vicini più prossimi;
* Si verifica se c'è una sola classe tra i k vicini più prossimi o se c'è un pareggio tra le occorrenze delle classi. In caso affermativo, si fa una scelta casuale tra le etichette coinvolte. Altrimenti, si seleziona l'etichetta più frequente;
* Infine, le previsioni vengono restituite sotto forma di un array NumPy;

# 3. Model	Evaluation


