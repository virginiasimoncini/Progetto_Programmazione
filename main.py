import sys
import numpy as np
import pandas as pd
from fetching import data_preprocessing
from knn import KNNClassifier
from evaluation import Holdout, XXCrossValidation, ModelEvaluator
import matplotlib.pyplot as plt

def holdout_validation(X, y, k_neighbors):
    # Chiedi all'utente di selezionare le metriche
    print("Scegli le metriche da validare:")
    print("1. Accuracy Rate")
    print("2. Error Rate")
    print("3. Sensitivity")
    print("4. Specificity")
    print("5. Geometric Mean")
    print("6. All the above")

    metric_choice = input("Inserisci il numero corrispondente all'opzione desiderata: ")

    if metric_choice == '6':
        metrics_to_validate = ['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean']
    else:
        metric_mapping = {
            '1': 'accuracy',
            '2': 'error_rate',
            '3': 'sensitivity',
            '4': 'specificity',
            '5': 'geometric_mean'
        }
        metrics_to_validate = [metric_mapping.get(metric_choice)]

    holdout_validation = Holdout(test_size=0.2)
    evaluator = ModelEvaluator(X, y, validation=holdout_validation, k=k_neighbors)
    evaluator.evaluate_validation(metrics=metrics_to_validate)

# Carica e pre-processa i dati
if len(sys.argv) > 1:
    X, y = data_preprocessing(file=sys.argv[1])
else:
    X, y = data_preprocessing()

# Chiedi all'utente di fornire il tipo di validazione
validation_type = input("Scegli il tipo di validazione (holdout/crossval): ").lower()
while validation_type not in ['holdout', 'crossval']:
    print("Scelta non valida. Scegli tra 'holdout' e 'crossval'.")
    validation_type = input("Scegli il tipo di validazione (holdout/crossval): ")

# Inizializza oggetto di validazione in base alla scelta dell'utente
if validation_type == 'holdout':
    # Chiedi all'utente di fornire il numero di vicini solo per Holdout
    k_neighbors = int(input("Inserisci il numero di vicini (k): "))
    holdout_validation(X, y, k_neighbors)
elif validation_type == 'crossval':
    # Chiedi all'utente di fornire il numero di folds solo per Cross Validation
    num_folds = int(input("Inserisci il numero di folds per la cross-validation: "))
    crossval_validation = XXCrossValidation(num_folds=num_folds)
    evaluator = ModelEvaluator(X, y, validation=crossval_validation, k=None)
    evaluator.evaluate_validation()
else:
    print("Tipo di validazione non riconosciuto. Inserisci 'holdout' o 'crossval'.")
    sys.exit(1)
