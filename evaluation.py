from fetching import data_preprocessing
from knn_model import knn_predict
import pandas as pd
import numpy as np
def evaluate_model(features, target, validation_type='Holdout', test_size=0.2):

    features, target = data_preprocessing()
    # Chiedo all'utente di inserire il valore di k
    k = int(input('Inserisci il valore di k per il modello KNN: '))
    
    if validation_type == 'Holdout':
        # Split del dataset in training e test set
        x_train, x_test, y_train, y_test = train_test(features, target, test_size=test_size, random_state=42)

        # Previsione utilizzando il KNN implementato in model_development
        predictions = [knn_predict(x_train, test_instance, k) for index, test_instance in x_test.iterrows()]

        # Calcolo delle metriche di valutazione
        accuracy = accuracy_score(y_test, predictions)
        error_rate = error_rate_manual(y_test, predictions)
        recall = recall_score_manual(y_test, predictions, pos_label=4)  # Assumendo da traccai 4 come maligno
        specificity = specificity_manual(y_test, predictions)
        g_mean = geometric_mean_manual(sensitivity=recall, specificity=specificity)

        # Stampo le metriche
        print(f"Accuracy: {accuracy}")
        print(f"Error Rate: {error_rate}")
        print(f"Recall: {recall}")
        print(f"Specificity: {specificity}")
        print(f"Geometric Mean: {g_mean}")

        # Salvo le metriche su un file Excel
        df_metrics_holdout = pd.DataFrame({
            'Accuracy': [accuracy],
            'Error Rate': [error_rate],
            'Recall': [recall],
            'Specificity': [specificity],
            'Geometric Mean': [g_mean]
        })
        df_metrics_holdout.to_excel("Risultati_evaluation_holdout.xlsx", index=False)

        # Restituisco le metriche
        return {
            'Accuracy': accuracy,
            'Error Rate': error_rate,
            'Recall': recall,
            'Specificity': specificity,
            'Geometric Mean': g_mean
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
            predictions = [knn_predict(train_features, test_instance, k) for _, test_instance in test_features.iterrows()]
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
