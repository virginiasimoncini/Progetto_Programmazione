import sys
import pandas as pd
from evaluation import ModelEvaluation
from fetching import data_preprocessing
import matplotlib.pyplot as plt

def main():
    # Load and preprocess dataset
    if len(sys.argv) > 1:
        X, y = data_preprocessing(file=sys.argv[1])
    else:
        X, y = data_preprocessing()
 
    # Chiede all'utente di scegliere il tipo di validazione
    validation_type = input("Choose validation type (holdout/xx): ").lower()
    while validation_type not in ['holdout', 'xx']:
        # Se la scelta non è valida, continua a chiedere fino a che non viene fornita una scelta valida
        print("Invalid choice. Choose between 'holdout' and 'xx'.")
        validation_type = input("Choose validation type (holdout/xx): ").lower()
 
    # Inizializza l'oggetto ModelEvaluation con i dati e il metodo di validazione scelto
    evaluator = ModelEvaluation(X, y, validation_type)
    # Chiede all'utente di inserire il numero di vicini per il KNN
    k_neighbors = int(input("k: "))
    
    # Inizializza una lista per aggregare i risultati delle metriche
    metrics_aggregated = []  # Lista per aggregare i risultati delle metriche

    # Gestisce la validazione holdout
    if validation_type == 'holdout':
        # Chiede la dimensione del test set
        test_size = float(input("Enter the test size: "))
        # Divide i dati in set di addestramento e test
        X_train, y_train, X_test, y_test = evaluator.split(X, y, test_size=test_size, random_state=None)
        # Valuta il modello sui dati di test
        df_evaluation = evaluator.evaluate(X_train, y_train, X_test, y_test, k_neighbors)
        #Aggiunge i risultati delle metriche alla lista
        metrics_aggregated.append(df_evaluation)  # Aggiunge il risultato in lista per consistenza
    else:  # LOOCV
        #Genera gli split per la LOOCV
        splits = evaluator.split(X, y, test_size=None, random_state=None)
        # Itera su ogni split e valuta il modello
        for split in splits:
            X_train, y_train, X_test, y_test = split
            df_evaluation = evaluator.evaluate(X_train, y_train, X_test, y_test, k_neighbors)
            metrics_aggregated.append(df_evaluation)
    
    # Calcola la media delle metriche per tutte le iterazioni
    df_evaluation_mean = pd.concat(metrics_aggregated).mean().to_frame().T
    
    # Scegli le metriche da visualizzare
    choice = select_metrics()
    
    # Stampa, salva e mostra le metriche
    evaluator.printMetrics(df_evaluation_mean, choice)
    evaluator.save_metrics(df_evaluation_mean)
    evaluator.metrics_plot(df_evaluation_mean)


def select_metrics():
    # Mappa le scelte delle metriche ai loro nomi descrittivi
    metrics_mapping = {
        '1': 'Accuracy Rate',
        '2': 'Error Rate',
        '3': 'Sensibility',
        '4': 'Specificity',
        '5': 'Geometric Mean',
        '6': 'All the above'
    }

    #Mostra le opzioni delle metriche all'utente
    print("Choose metrics to validate:")
    for key, value in metrics_mapping.items():
        # Itera attraverso il dizionario delle metriche e stampa ogni opzione disponibile.
        print(f"{key}. {value}")

    # Richiede all'utente di inserire il numero corrispondente all'opzione desiderata.
    metric_choice = input("Enter the number corresponding to the desired option: ")

    # Controlla se la scelta dell'utente è presente nel dizionario delle metriche.
    if metric_choice not in metrics_mapping:
        # Se la scelta non è valida (non è una chiave nel dizionario), stampa un messaggio di errore.
        print("Invalid choice. Exiting program.")
        sys.exit()  # Termina il programma se la scelta non è valida.

    # Restituisce la chiave numerica (come stringa) corrispondente alla scelta dell'utente.
    return metric_choice
 
if __name__ == "__main__":
    main()

