import sys
import numpy as np
import pandas as pd
from fetching import data_preprocessing
from knn import KNNClassifier
from evaluation import Holdout, XXCrossValidation, ModelEvaluator
import matplotlib.pyplot as plt

# Carica e pre-processa i dati
if len(sys.argv) > 1:
    X, y = data_preprocessing(file=sys.argv[1])
else:
    X, y = data_preprocessing()

# Chiedi all'utente di fornire il tipo di validazione
validation_type = input("Scegli il tipo di validazione (holdout/crossval): ").lower()
while validation_type not in ['holdout', 'crossval']:
    print("Scelta non valida. Scegli tra 'holdout' e 'crossval'.")
    validation_type = input("Scegli il tipo di validazione (holdout/crossv")

# Inizializza oggetto di validazione in base alla scelta dell'utente
if validation_type == 'holdout':
    # Chiedi all'utente di fornire il numero di vicini solo per Holdout
    k_neighbors = int(input("Inserisci il numero di vicini (k): "))
    holdout_validation = Holdout(test_size=0.2)
    evaluator = ModelEvaluator(X, y, validation=holdout_validation, k=k_neighbors)
elif validation_type == 'crossval':
    # Chiedi all'utente di fornire il numero di folds solo per Cross Validation
    num_folds = int(input("Inserisci il numero di folds per la cross-validation: "))
    crossval_validation = XXCrossValidation(num_folds=num_folds)
    evaluator = ModelEvaluator(X, y, validation=crossval_validation, k=None)
else:
    print("Tipo di validazione non riconosciuto. Inserisci 'holdout' o 'crossval'.")
    sys.exit(1)

# Esegui la validazione
evaluator.evaluate_validation()
