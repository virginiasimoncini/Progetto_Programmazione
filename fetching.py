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

    # Fixing missing values appropiately (We only analyse the features given that every instance must have a target):
    x.bfill()
    print(x.head())
    print(y.head())
    #return the targets and features
    return x,y


# Definisco una funzione per valutare un modello di classificazione
def evaluate_model(features, target, validation_type='Holdout', test_size=0.2, k=5, metrics=['accuracy']):
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

    # Seleziona il tipo di validazione
    if validation_type == 'Holdout':
        # Holdout validation
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

        # Crea il modello K-NN con n_neighbors=k numero di vicini k
        model = KNeighborsClassifier(n_neighbors=k)

        # Addestra il modello sul set di addestramento, il .fit mette in relazione
        model.fit(x_train, y_train)

        # Genera le previsioni sul set di test
        predictions = model.predict(x_test)

    elif validation_type == 'XX':
        # K-Fold Cross Validation
        model = KNeighborsClassifier(n_neighbors=k)

        # Calcola i punteggi di Cross Validation
        scores = cross_val_score(model, features, target, cv=5)  # 5-fold cross validation

        # Stampa i punteggi e la media
        print("Cross Validation Scores:", scores)
        print("Mean Accuracy:", np.mean(scores))

        # Restituisce la media dei punteggi di Cross Validation
        return np.mean(scores)

    # Calcola le metriche specificate
    evaluation_metrics = {}
    if 'accuracy' in metrics:
        accuracy = accuracy_score(y_test, predictions)
        evaluation_metrics['Accuracy'] = accuracy
    if 'sensitivity' in metrics:
        sensitivity = recall_score(y_test, predictions)
        evaluation_metrics['Sensitivity'] = sensitivity

    # Stampa e restituisci le metriche di valutazione
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    return evaluation_metrics
