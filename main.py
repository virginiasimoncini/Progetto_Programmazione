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
    validation_type = input("Scegli il tipo di validazione (holdout/crossval): ")

# Inizializza oggetto di validazione in base alla scelta dell'utente
if validation_type == 'holdout':
    # Chiedi all'utente di fornire il numero di vicini solo per Holdout
    k_neighbors = int(input("Inserisci il numero di vicini (k): "))
    validation = Holdout(test_size=0.2)
    evaluator = ModelEvaluator(X, y, validation=validation, k=k_neighbors)

    print("Scegli la metrica da validare:")
    print("1. Accuracy Rate")
    print("2. Error Rate")
    print("3. Specificity")
    print("4. Geometric Mean")
    print("5. Sensitivity")
    print("6. Tutte le Metriche")

    metric_choice = input("Inserisci il numero corrispondente all'opzione desiderata: ")

    # Seleziona tutte le metriche se l'utente sceglie 5
    if metric_choice == '6':
        metrics_to_validate = ['accuracy', 'error_rate', 'specificity', 'geometric_mean','sensitivity']
    else:
        # Mappa la scelta dell'utente a una metrica
        metric_mapping = {
            '1': 'accuracy',
            '2': 'error_rate',
            '3': 'specificity',
            '4': 'geometric_mean',
            '5': 'sensibility',
        }
        selected_metric = metric_mapping.get(metric_choice)
        metrics_to_validate = [selected_metric]

    # Valida le metriche selezionate
    evaluator.evaluate_validation(metrics=metrics_to_validate)

else:
    num_folds = int(input("Inserisci il numero di folds per la cross-validation: "))
    validation = XXCrossValidation(num_folds=num_folds)
    evaluator = ModelEvaluator(X, y, validation=validation,k=None)

    # Chiedi all'utente di selezionare la metrica (per ora solo accuracy in cross-validation)
    evaluator.evaluate_validation()
