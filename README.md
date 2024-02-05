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

1. **Data	Preprocessing:**
This branch process the dataset.Its steps are:
    - **Load the dataset:**
        We import the dataset from a file called `breast_cancer.csv`
    - **Divide dataset in features and targets:**
We pop the "Class" column from the dataframe and assign it to the Y variable. Then we assign the rest of the dataframe to the X variable.
    - **Process missing values appropiately:**
We must take into account the missing values in the features dataset. For this we have used the "blackfill" method.
2. **Model	Development**

3. **Model	Evaluation:**

# Data preprocessing
This branch process the dataset.Its steps are:
### Load the dataset:
We import the dataset from a file called `breast_cancer.csv`
### Divide dataset in features and targets
We pop the "Class" column from the dataframe and assign it to the Y variable. Then we assign the rest of the dataframe to the X variable.
### Process missing values appropiately:
We must take into account the missing values in the features dataset. For this we have used the "blackfill" method.
