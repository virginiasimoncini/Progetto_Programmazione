import sys
import numpy as np
import pandas as pd
from fetching import data_preprocessing
from knn import KNNClassifier
from evaluation import Holdout, XXCrossValidation, ModelEvaluator
import matplotlib.pyplot as plt

# Carica e pre-processa i dati
if len(sys.argv) > 1 :
    X, y = data_preprocessing(file = sys.argv[1])
else:
    X, y = data_preprocessing()

# Definisci le opzioni di validazione incapsulate in oggetti Holdout e XXCrossValidation
holdout_validation = Holdout(test_size=0.2, random_state=42)

# Chiedi all'utente di fornire il numero di vicini
k_neighbors = int(input("Inserisci il numero di vicini (k): "))

# Puoi gestire le ecc con un while se necessario sia per la valid_type che per il k per catturare l'excp

# Puoi inserire un numero n di quanti "miniset" gestire

num_folds = int(input("Inserisci il numero di folds per la cross-validation: "))

# Crea l'oggetto XXCrossValidation senza specificare il valore di k
crossval_validation = XXCrossValidation(num_folds=num_folds)



# Chiedi all'utente di fornire il tipo di validazione
validation_type = input("Scegli il tipo di validazione (holdout/crossval): ").lower()

# Esegui la validazione
if validation_type == 'holdout':
    evaluator = ModelEvaluator(X, y, validation=holdout_validation, k=k_neighbors)
elif validation_type == 'crossval':
    evaluator = ModelEvaluator(X, y, validation=crossval_validation, k=None)
else:
    raise ValueError("Tipo di validazione non riconosciuto. Inserisci 'holdout' o 'crossval'.")

# Esegui la validazione
evaluator.evaluate_validation()
