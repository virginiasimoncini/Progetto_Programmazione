import os
import numpy as np
import pandas as pd
from fetching import data_preprocessing
from knn import KNNClassifier
import matplotlib.pyplot as plt

class Holdout:
    def __init__(self, test_size, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df):
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(df))
        test_set_size = int(len(df) * self.test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        train_set = df.iloc[train_indices]
        test_set = df.iloc[test_indices]

        return train_set, test_set

class XXCrossValidation:
    def __init__(self, num_folds):
        """
        Inizializza l'oggetto XXCrossValidation.

        Parameters:
        - num_folds (int): Numero di fold nella cross-validation.

        """
        self.num_folds = num_folds

    def split(self, df):
        """
        Suddivide il DataFrame in fold per la cross-validation.

        Parameters:
        - df (pd.DataFrame): Il DataFrame contenente i dati.

        Returns:
        - folds (list): Una lista di tuple, ognuna contenente un set di addestramento e un set di test.

        """
        folds = []

        # Imposta il seed per la riproducibilità
        np.random.seed(42)

        # Ottiene gli indici del DataFrame in modo casuale
        indices = np.random.permutation(df.index)

        # Mescola le righe del DataFrame
        features_mixed = df.loc[indices]

        # Calcola la lunghezza di ciascun fold
        fold_length = len(df) // self.num_folds

        # Itera su ogni fold
        for j in range(self.num_folds):
            # Calcola gli indici di inizio e fine del fold
            test_start, test_end = j * fold_length, (j + 1) * fold_length 
            
            # Ottiene il set di test
            test_set = features_mixed.iloc[test_start:test_end]
            
            # Ottiene il set di addestramento concatenando le parti del DataFrame escludendo il fold di test
            train_set = pd.concat([features_mixed.iloc[:test_start], features_mixed.iloc[test_end:]])
            
            # Aggiunge la tupla al risultato
            folds.append((train_set, test_set))

        return folds




class ModelEvaluator:
    def __init__(self, X, y, validation, k=None):
        self.X = X
        self.y = y
        self.validation = validation
        self.k = k

    def evaluate(self, X_train, X_test, y_train, y_test):
        knn = KNNClassifier(k=self.k)
        knn.fit(X_train, y_train)

        # Utilizzo del metodo predict aggiornato che gestisce sia DataFrame che array NumPy
        if isinstance(X_test, pd.DataFrame):
            y_pred = knn.predict(X_test)
        else:
            y_pred = knn.predict(X_test.values)

        accuracy = (y_pred == y_test.values).mean()
        error_rate = 1 - accuracy

        # Calcolo della specificità
        true_negative = np.sum((y_pred == 2) & (y_test == 2))
        false_positive = np.sum((y_pred == 4) & (y_test == 2))
        specificity = true_negative / (true_negative + false_positive)

        # Calcolo della geometric mean
        recall = np.sum((y_pred == 4) & (y_test == 4)) / np.sum(y_test == 4)
        g_mean = np.sqrt(specificity * recall)

        return accuracy, error_rate, specificity, g_mean

    # ...

def evaluate_validation(self):
    if not os.path.exists("output"):
        os.makedirs("output")
    
    if isinstance(self.validation, Holdout):
            train_set, test_set = self.validation.split(pd.concat([self.X, self.y], axis=1))
            accuracy, error_rate, specificity, g_mean = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])

            print("Holdout Evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Error Rate: {error_rate:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"Geometric Mean: {g_mean:.4f}")

            # Salvataggio dei risultati in Excel
            results_df = pd.DataFrame({
                'Validation Type': ['Holdout'],
                'Accuracy': [accuracy],
                'Error Rate': [error_rate],
                'Specificity': [specificity],
                'Geometric Mean': [g_mean]
            })
            results_df.to_excel('output/validation_results.xlsx', index=False)

            # Plot delle performance
            plt.bar(['Accuracy', 'Error Rate', 'Specificity', 'Geometric Mean'], [accuracy, error_rate, specificity, g_mean])
            plt.title('Holdout Evaluation Metrics')
            plt.ylabel('Metric Value')
            plt.savefig('output/holdout_evaluation_plot.png')
            plt.show()
            
    elif isinstance(self.validation, XXCrossValidation):
        accuracies, error_rates, specificities, g_means = [], [], [], []
        
        # Aggiunta di una gestione speciale per il caso di cross-validation
        validation_type_list = [f'Fold {i + 1}' for i in range(self.k)] if self.k is not None else ['Mean']
        
        for i, (train_set, test_set) in enumerate(self.validation.split(pd.concat([self.X, self.y], axis=1))):
            accuracy, error_rate, specificity, g_mean = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])
            accuracies.append(accuracy)
            error_rates.append(error_rate)
            specificities.append(specificity)
            g_means.append(g_mean)

            print(f"Fold {i + 1} Accuracy: {accuracy:.4f}")

        mean_accuracy = np.mean(accuracies)

        print(f"\nMean Accuracy across {self.k} folds: {mean_accuracy:.4f}")

        # Salvataggio dei risultati in Excel
        results_df = pd.DataFrame({
            'Validation Type': validation_type_list,
            'Accuracy': accuracies + [mean_accuracy],
            'Error Rate': error_rates + [np.mean(error_rates)],
            'Specificity': specificities + [np.mean(specificities)],
            'Geometric Mean': g_means + [np.mean(g_means)]
        })
        results_df.to_excel('output/validation_results.xlsx', index=False)

        # Plot delle performance
        for metric in ['Accuracy', 'Error Rate', 'Specificity', 'Geometric Mean']:
            plt.plot(results_df['Validation Type'], results_df[metric], label=metric)

        plt.title('Cross-Validation Evaluation Metrics')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig('output/crossval_evaluation_plot.png')
        plt.show()

    elif isinstance(self.validation, XXCrossValidation):
            accuracies, error_rates, specificities, g_means = [], [], [], []
            for i, (train_set, test_set) in enumerate(self.validation.split(pd.concat([self.X, self.y], axis=1))):
                accuracy, error_rate, specificity, g_mean = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])
                accuracies.append(accuracy)
                error_rates.append(error_rate)
                specificities.append(specificity)
                g_means.append(g_mean)

                print(f"Fold {i + 1} Accuracy: {accuracy:.4f}")

            mean_accuracy = np.mean(accuracies)

            print(f"\nMean Accuracy across {self.k} folds: {mean_accuracy:.4f}")

            # Salvataggio dei risultati in Excel
            results_df = pd.DataFrame({
                'Validation Type': [f'Fold {i + 1}' for i in range(self.k)] + ['Mean'],
                'Accuracy': accuracies + [mean_accuracy],
                'Error Rate': error_rates + [np.mean(error_rates)],
                'Specificity': specificities + [np.mean(specificities)],
                'Geometric Mean': g_means + [np.mean(g_means)]
            })
            results_df.to_excel('output/validation_results.xlsx', index=False)

            # Plot delle performance
            for metric in ['Accuracy', 'Error Rate', 'Specificity', 'Geometric Mean']:
                plt.plot(results_df['Validation Type'], results_df[metric], label=metric)

            plt.title('Cross-Validation Evaluation Metrics')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.savefig('output/crossval_evaluation_plot.png')
            plt.show()
