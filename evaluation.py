import os
import numpy as np
import pandas as pd
from fetching import data_preprocessing
from knn import KNNClassifier
import matplotlib.pyplot as plt

metric_mapping = [
    'Accuracy',
    'Error Rate',
    'Specificity',
    'Geometric Mean',
    'Sensibility',
]

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
        self.num_folds = num_folds

    def split(self, df):
        folds = []
        np.random.seed(42)
        indices = np.random.permutation(df.index)
        features_mixed = df.loc[indices]
        fold_length = len(df) // self.num_folds

        for j in range(self.num_folds):
            test_start, test_end = j * fold_length, (j + 1) * fold_length 
            test_set = features_mixed.iloc[test_start:test_end]
            train_set = pd.concat([features_mixed.iloc[:test_start], features_mixed.iloc[test_end:]])
            folds.append((train_set, test_set))

        return folds

class ModelEvaluator:
    def __init__(self, X, y, validation, k=None, num_folds=None):
        self.X = X
        self.y = y
        self.validation = validation
        self.k = k
        self.num_folds = num_folds
        print("Scegli la metrica da validare:")
        print("1. Accuracy Rate")
        print("2. Error Rate")
        print("3. Specificity")
        print("4. Geometric Mean")
        print("5. Sensitivity")
        print("6. Tutte le Metriche")
        self.choice = int(input("Inserisci il numero corrispondente all'opzione desiderata: "))
    
    #keep it
    def evaluate(self, X_train, X_test, y_train, y_test):
        knn = KNNClassifier(k=self.k)
        knn.fit(X_train, y_train)
        #create a switch for the choice 
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
        if (true_negative + false_positive) != 0:
           specificity = true_negative / (true_negative + false_positive)
        else:
            specificity = 0  # o un altro valore appropriato


        true_positive = np.sum((y_pred == 4) & (y_test == 4))
        sensitivity = true_positive / np.sum(y_test == 4)

        # Calcolo della geometric mean
        recall = np.sum((y_pred == 4) & (y_test == 4)) / np.sum(y_test == 4)
        g_mean = np.sqrt(specificity * recall)

        return accuracy, error_rate, specificity, g_mean, sensitivity
    #dont
    def evaluate_validation(self,metrics):
        if not os.path.exists("output"):
            os.makedirs("output")
        if isinstance(self.validation, Holdout):
            train_set, test_set = self.validation.split(pd.concat([self.X, self.y], axis=1))
            result = self.evaluate_and_store_results(train_set, test_set,metrics)
        else:
            for i, (train_set, test_set) in enumerate(self.validation.split(pd.concat([self.X, self.y], axis=1))):
                result = self.evaluate_and_store_results(train_set, test_set, metrics, i)

        self.calculate_and_print_mean(result,metrics)

        #result['Validation Type'] = ['Mean']

        # Save results to Excel
        self.save_to_excel(result)

        # Plot delle performance
        self.plot_results(result)
    #dont
    def evaluate_and_store_results(self, train_set, test_set, metrics, fold_number=None):
        accuracy, error_rate, specificity, g_mean, sensitivity = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])
        results=[accuracy,error_rate,specificity,g_mean,sensitivity]

        if fold_number is not None:
            print(f"{self.validation.__class__.__name__} {fold_number + 1} {metric_mapping[metrics]}: {accuracy:.4f}")
        else:
            print(f"{self.validation.__class__.__name__} {metric_mapping[metrics]}: {accuracy:.4f}")
        return results[metrics]

    def calculate_and_print_mean(self, results,metrics):
        mean_accuracy = np.mean(results)

        print(f"\nMean Evaluation across {self.validation.__class__.__name__}:")
        print(f"Mean {metric_mapping[metrics]}: {mean_accuracy:.4f}")

    def save_to_excel(self, results):
        # Verifica e allinea le lunghezze delle colonne
        lengths = [len(results[col]) for col in results.keys()]
        max_length = max(lengths)

        for col in results.keys():
            current_length = len(results[col])
            if current_length < max_length:
                results[col] += [np.nan] * (max_length - current_length)

        # Crea il DataFrame
        results_df = pd.DataFrame(results)

        results_df.to_excel('output/validation_results.xlsx', index=False)

    def plot_results(self, results):
        # Plot delle performance
        metrics = ['Accuracy', 'Error Rate', 'Specificity', 'Sensitivity', 'Geometric Mean']
        plt.bar(metrics, results['Accuracy'])  # Modifica qui per la metrica che vuoi visualizzare
        plt.title(f'{self.validation.__class__.__name__} Evaluation Metrics')
        plt.ylabel('Metric Value')
        plt.savefig(f'output/{self.validation.__class__.__name__.lower()}_evaluation_plot.png')
        plt.show()