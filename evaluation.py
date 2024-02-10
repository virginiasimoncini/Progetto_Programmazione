# Classe ModelEvaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier  # Assicurati di avere l'import corretto per il tuo algoritmo KNN

class ModelEvaluation:
    def __init__(self, features: pd.DataFrame, target: pd.Series, k: int, evaluation_method: str, params: dict,
                 chosen_metrics: list):
        self.features = features
        self.target = target
        self.k = k
        self.evaluation_method = evaluation_method
        self.params = params
        self.chosen_metrics = chosen_metrics

    def evaluate_model(self):
        if self.evaluation_method == 'holdout':
            return self.holdout_validation()
        elif self.evaluation_method == 'xx':
            return self.cross_validation()

    def holdout_validation(self):
        training_perc = self.params.get('training_perc', 70)
        X_train, y_train, X_test, y_test = self.split(self.features, self.target, training_perc)

        knn_model = KNNClassifier(self.k, X_train, y_train)
        predictions = knn_model.model_prediction(X_test)

        accuracy, error_rate, sensibility, specificity, geometric_mean = self.__metrics_calculation(y_test, predictions)

        return accuracy, error_rate, sensibility, specificity, geometric_mean

    def cross_validation(self):
        k_iterations = self.params.get('k_iterations', 5)

        accuracy_scores = []
        error_rate_scores = []
        sensibility_scores = []
        specificity_scores = []
        geometric_mean_scores = []

        for _ in range(k_iterations):
            X_train, y_train, X_test, y_test = self.split(self.features, self.target, 70)

            knn_model = KNNClassifier(self.k, X_train, y_train)
            predictions = knn_model.model_prediction(X_test)

            accuracy, error_rate, sensibility, specificity, geometric_mean = self.__metrics_calculation(y_test, predictions)

            accuracy_scores.append(accuracy)
            error_rate_scores.append(error_rate)
            sensibility_scores.append(sensibility)
            specificity_scores.append(specificity)
            geometric_mean_scores.append(geometric_mean)

        accuracy_mean = np.mean(accuracy_scores)
        error_rate_mean = np.mean(error_rate_scores)
        sensibility_mean = np.mean(sensibility_scores)
        specificity_mean = np.mean(specificity_scores)
        geometric_mean_mean = np.mean(geometric_mean_scores)

        return accuracy, error_rate, sensibility, specificity, geometric_mean


    def split(self, df, target_column, test_size=0.3, random_state=None):
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(df))
        test_set_size = int(len(df) * test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_size:]

        train_set = df.iloc[train_indices]
        test_set = df.iloc[test_indices]

        # Separare le etichette dai dati
        X_train = train_set.drop(target_column, axis=1)
        y_train = train_set[target_column]

        X_test = test_set.drop(target_column, axis=1)
        y_test = test_set[target_column]

        return X_train, y_train, X_test, y_test

    def __metrics_calculation(self, y_test: pd.Series, predictions: list):
        # Inizializza le variabili che conterranno i valori delle metriche calcolate
        accuracy= 0
        error_rate = 0
        sensibility = 0
        specificity = 0
        geometric_mean = 0

        # Calcolo della matrice di confusione
        true_negative = np.sum((y == pred and pred == 2) for y, pred in zip(y_test.values, predictions))
        true_positive = np.sum((y == pred and pred == 4) for y, pred in zip(y_test.values, predictions))
        false_positive = np.sum((y != pred and pred == 4) for y, pred in zip(y_test.values, predictions))
        false_negative = np.sum((y != pred and pred == 2) for y, pred in zip(y_test.values, predictions))

        # Calcolo effettivo delle metriche richieste utilizzando i valori della matrice di confusione precedentemente calcolati
        if len(y_test) > 0:
            accuracy= (true_negative + true_positive) / len(y_test)  # Percentuale di predizioni corrette rispetto al totale delle predizioni
        if len(y_test) > 0:
            error_rate = (false_positive + false_negative) / len(y_test)  # Percentuale di predizioni errate rispetto al totale delle predizioni
        if (true_positive + false_negative) != 0:
            sensibility = true_positive / (true_positive + false_negative)  # Abilità del modello di predire correttamente i valori positivi
        if (true_negative + false_positive) != 0:
            specificity = true_negative / (true_negative + false_positive)  # Abilità del modello di predire correttamente i valori negativi
        if sensibility  != 0 and specificity != 0:
            geometric_mean = np.sqrt(sensibility  * specificity)  # Media che bilancia i valori positivi e negativi

        return accuracy, error_rate, sensibility , specificity, geometric_mean

    def __save_metrics(self, metrics_list: list):
        metrics_transposed = list(map(list, zip(*metrics_list)))
        metrics_df = pd.DataFrame(metrics_transposed, columns=['Accuracy', 'Error Rate', 'Sensibility', 'Specificity', 'Geometric Mean'])
      
        metrics_df.to_excel('Metrics.xlsx', index=False)

    def __metrics_plot(self, accuracy: list, error_rate: list, sensibility: list, specificity: list, geometric_mean: list):
       labels = ['Accuracy', 'Error Rate', 'Sensibility', 'Specificity', 'Geometric Mean']
       values = [accuracy, error_rate, sensibility, specificity, geometric_mean]

     # Creazione del grafico delle metriche nel tempo
       plt.figure(figsize=(10, 5))

       for label, value in zip(labels, values):
          plt.plot(value, marker='o', linestyle='solid', linewidth=2, markersize=5, label=label)

       plt.legend(loc='upper right')
       plt.xlabel("Esperimenti")
       plt.ylabel("Valori")
       plt.title("Andamento delle Metriche")
       plt.tight_layout()

    # Salva il grafico come un'immagine
       plt.savefig('metrics_plot.png')

    # Creazione di un nuovo grafico per il boxplot delle metriche
       plt.figure(figsize=(10, 5))
       plt.boxplot(values)

       plt.xlabel("Metriche")
       plt.ylabel("Valori")
       plt.title("Boxplot delle Metriche")
       plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
       plt.tight_layout()

    # Salva il boxplot come un'immagine
    plt.savefig('boxplot_metrics.png')

    # Chiudi i grafici per evitare di visualizzarli in modo interattivo (se necessario)
    plt.close('all')
