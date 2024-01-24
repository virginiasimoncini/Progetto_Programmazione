import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
import numpy as np


def data_preprocessing():
    # fetch dataset 
    df = pd.read_csv("breast_cancer.csv") 
  
    # data (as pandas dataframes)
    y = df.pop("Class")
    x = df
    # normalize features
    for column in x.columns: 
        x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())       

    # Fixing missing values appropriately (We only analyze the features given that every instance must have a target):
    x.bfill()
    print(x.head())
    print(y.head())
    # return the targets and features
    return x, y


# Definisco una funzione per valutare un modello di classificazione
# lo 02.2 indica che stai eseguendo la validazione Holdout con il 20% dei dati nel set di test, nel caso si può mettere come input per scegliere. 
def evaluate_model(features, target, validation_type='Holdout', test_size=0.2, metrics=['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean']):
    """
    Questa funzione valuta un modello di classificazione utilizzando Holdout o K-Fold Cross Validation.

    Parametri:
    - features: Variabili indipendenti (X).
    - target: Variabile dipendente o label (y).
    - validation_type: Tipo di validazione ('Holdout' o 'XX' per Cross Validation).
    - test_size: Percentuale di dati da utilizzare nei set di test (solo per Holdout).
    - k: Numero di vicini da utilizzare per il classificatore K-NN.
    - metrics: Lista delle metriche di valutazione da calcolare.

    Restituisce:
    - Se validation_type è 'Holdout', restituisce un dizionario contenente le metriche calcolate.
    - Se validation_type è 'XX', stampa i punteggi di Cross Validation e restituisce la media di essi.
    """
    validation_type = input("Inserisci il tipo di validazione ('Holdout' o 'XX' per Cross Validation): ") 

     # Se l'utente mi inserisce un input che non sia tra i due che ho proposto io il codice mi deve "ciclare" finche non ho
    # l'input giusto per iniziare il processo

    while validation_type  != 'Holdout' and validation_type != 'XX':
        validation_type = input("Input non valido. Inserisci il tipo di validazione ('Holdout' o 'XX' per Cross Validation): ")


    # Seleziona il tipo di validazione
    # il validation type può essere passato come argomento alla funzione evaluate_model
    if validation_type == 'Holdout':
        # Holdout validation
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
        k = int(input("Inserisci il numero di vicini (k): "))

        # Crea il modello K-NN con n_neighbors=k numero di vicini k con cui fare il "confronto" per unire
        model = KNeighborsClassifier(n_neighbors=k)

        # Addestra il modello sul set di addestramento, il .fit mette in relazione
        model.fit(x_train, y_train)

        # Genera le previsioni sul set di test
        predictions = model.predict(x_test)

        # Calcola le metriche specificate
        evaluation_metrics = {}
        # L'accuracy rappresenta la percentuale di predizioni corrette rispetto al totale delle predizioni.
        if 'accuracy' in metrics:
            accuracy = accuracy_score(y_test, predictions)
            evaluation_metrics['Accuracy'] = accuracy
        # L'error rate rappresenta la percentuale di predizioni errate rispetto al totale delle predizioni.
        if 'error_rate' in metrics:
            error_rate = 1 - accuracy
            evaluation_metrics['Error Rate'] = error_rate
        # La sensitivity rappresenta la percentuale di positivi veri correttamente identificati rispetto al totale dei positivi veri.
        if 'sensitivity' in metrics:
            sensitivity = recall_score(y_test, predictions)
            evaluation_metrics['Sensitivity'] = sensitivity
        # La specificity rappresenta la percentuale di negativi veri correttamente identificati rispetto al totale dei negativi veri.
        #ravel() anzichè darmi la matrice mi da un vettore unico che contiene la matrice
            
        if 'specificity' in metrics:
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            specificity = tn / (tn + fp)
            evaluation_metrics['Specificity'] = specificity


        # Il geometric mean è la radice quadrata del prodotto di sensitivity e specificity.
        if 'geometric_mean' in metrics:
            geometric_mean = np.sqrt(sensitivity * specificity)
            evaluation_metrics['Geometric Mean'] = geometric_mean



        # Stampa e restituisci le metriche di valutazione iterando e restituendo chiave:valore
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value}")
    
                # Aggiungi le metriche al DataFrame
        df_metrics_holdout = pd.DataFrame(evaluation_metrics, index=[0])

        # Salva il DataFrame su un file Excel
        df_metrics_holdout.to_excel("Risultati_evaluation_holdout.xlsx", index=False)

        return evaluation_metrics



    elif validation_type == 'XX':
        # K-Fold Cross Validation
        model = KNeighborsClassifier(n_neighbors=k)

        # Calcola i punteggi di Cross Validation
        scores = cross_val_score(model, features, target, cv=5)  # 5-fold cross validation

        # Stampa i punteggi e la media
        print("Cross Validation Scores:", scores)
        print("Mean Accuracy:", np.mean(scores))

        # Aggiungi la media al DataFrame
        df_metrics_cv = pd.DataFrame({'Mean Accuracy': [(np.scores)]})

        # Salva il DataFrame su un file Excel
        df_metrics_cv.to_excel("Risultati_evaluation_cross_validation.xlsx", index=False)

        # Restituisce la media dei punteggi di Cross Validation
        return np.mean(scores)

