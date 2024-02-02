from fetching import data_preprocessing
from knn import KNNClassifier, knn_predict, predict_most_common_label
import pandas as pd
import numpy as np

#Calcola la percentuale di predizioni corrette fatte dal modello
def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

#Calcola il tasso di errore del modello, che rappresenta la frazione di predizioni errate rispetto al totale delle predizioni.
def error_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

#calcola il tasso di richiamo Del modello,
#che misura la capacità del modello di identificare correttamente tutti gli esempi positivi.
def recall_score(y_true, y_pred, pos_label):
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    recall = true_positives / actual_positives
    return recall

#Calcola la specificità del modello. 
#La specificità rappresenta la capacità del modello di identificare correttamente tutti gli esempi negativi.
def specificity(y_true, y_pred):
    true_negatives = np.sum((y_true != 4) & (y_pred != 4))
    actual_negatives = np.sum(y_true != 4)
    specificity = true_negatives / actual_negatives
    return specificity

#Calcola la media geometrica tra sensibilità (recall) e specificità.
def geometric_mean(sensitivity, specificity):
    g_mean = np.sqrt(sensitivity * specificity)
    return g_mean

def custom_train_test_split(features, target, test_size=0.2, random_state=None):
    np.random.seed(random_state)  # Imposta il seed per la riproducibilità
    indices = np.random.permutation(len(features))

    test_size = int(test_size * len(features))
    test_indices, train_indices = indices[:test_size], indices[test_size:]

    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

    return x_train, x_test, y_train, y_test

def evaluate_model(features, target, validation_type='Holdout', test_size=0.2):
    features, target = data_preprocessing()
    k = int(input('Inserisci il valore di k per il modello KNN: '))
    
    if validation_type == 'Holdout':
        # Split del dataset in training e test set utilizzando la nuova funzione custom_train_test_split
        x_train, x_test, y_train, y_test = custom_train_test_split(features, target, test_size=test_size, random_state=42)

        # Previsione utilizzando il KNN implementato in model_development
        knn_classifier = KNNClassifier(k)
        knn_classifier.fit(x_train, y_train)
        predictions = [knn_predict(x_train, test_instance, k) for index, test_instance in x_test.iterrows()]

        # Calcolo delle metriche di valutazione
        accuracy = accuracy_score(y_test, predictions)
        error_rate_val = error_rate(y_test, predictions)
        recall_val = recall_score(y_test, predictions, pos_label=4)  # Assumendo da traccai 4 come maligno
        specificity_val = specificity(y_test, predictions)
        g_mean_val = geometric_mean(sensitivity=recall_val, specificity=specificity_val)

        # Stampo le metriche
        print(f"Accuracy: {accuracy}")
        print(f"Error Rate: {error_rate_val}")
        print(f"Recall: {recall_val}")
        print(f"Specificity: {specificity_val}")
        print(f"Geometric Mean: {g_mean_val}")

        # Salvo le metriche su un file Excel
        df_metrics_holdout = pd.DataFrame({
            'Accuracy': [accuracy],
            'Error Rate': [error_rate_val],
            'Recall': [recall_val],
            'Specificity': [specificity_val],
            'Geometric Mean': [g_mean_val]
        })
        df_metrics_holdout.to_excel("Risultati_evaluation_holdout.xlsx", index=False)

        # Restituisco le metriche
        return {
            'Accuracy': accuracy,
            'Error Rate': error_rate_val,
            'Recall': recall_val,
            'Specificity': specificity_val,
            'Geometric Mean': g_mean_val
        }
    
    elif validation_type == 'XX':
        # K-Fold Cross Validation
        np.random.seed(42)

        # Permutazione casuale degli indici
        indices = np.random.permutation(features.index)
        features_mixed = features.loc[indices]
        target_mixed = target.loc[indices]

        scores = []

        fold_length = len(features) // k

        for i in range(k):
            # Divisione del dataset in training e test set per ogni fold
            test_start, test_end = i * fold_length, (i + 1) * fold_length
            test_features, test_target = features_mixed.iloc[test_start:test_end], target_mixed.iloc[test_start:test_end]
            train_features = pd.concat([features_mixed.iloc[:test_start], features_mixed.iloc[test_end:]])
            train_target = pd.concat([target_mixed.iloc[:test_start], target_mixed.iloc[test_end:]])

            # Predizione utilizzando il KNN
            predictions = [knn_predict(train_features, test_instance, k) for index, test_instance in test_features.iterrows()]
            fold_accuracy = accuracy_score(test_target, predictions)
            scores.append(fold_accuracy)

        # Calcolo della media delle accuratezze per i fold
        mean_accuracy = np.mean(scores)
        print("Mean Accuracy:", mean_accuracy)

        # Salvo la media delle accuratezze su un file Excel
        df_metrics_cv = pd.DataFrame({'Mean Accuracy': [mean_accuracy]})
        df_metrics_cv.to_excel("Risultati_evaluation_cross_validation.xlsx", index=False)

        # Restituisco la media delle accuratezze
        return mean_accuracy
