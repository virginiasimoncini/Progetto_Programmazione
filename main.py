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

# Chiedi all'utente di fornire il numero di vicini
k_neighbors = int(input("Inserisci il numero di vicini (k): "))

# Inizializza le variabili degli oggetti a None
holdout_validation = None
crossval_validation = None

# Chiedi all'utente di fornire il tipo di validazione
validation_type = input("Scegli il tipo di validazione (holdout/crossval): ").lower()

# In base alla scelta faccio l'istanziazione degli oggetti solo se necessario
if validation_type == 'holdout':
    holdout_validation = Holdout()
elif validation_type == 'crossval':
    num_folds = int(input("Inserisci il numero di folds per la cross-validation: "))
    crossval_validation = XXCrossValidation(num_folds=num_folds)
else:
    raise ValueError("Tipo di validazione non riconosciuto. Inserisci 'holdout' o 'crossval'.")

# Esegui la validazione solo se uno degli oggetti è stato inizializzato
if holdout_validation or crossval_validation:
    evaluator = ModelEvaluator(X, y, validation=holdout_validation or crossval_validation, k=k_neighbors)
    evaluator.evaluate_validation()
