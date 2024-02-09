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

    def evaluate(self, X_train, X_test, y_train, y_test):
        knn = KNNClassifier(k=self.k)
        knn.fit(X_train, y_train)

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

    def evaluate_validation(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        results = {'Accuracy': [], 'Error Rate': [], 'Specificity': [], 'Sensitivity': [], 'Geometric Mean': []}

        if isinstance(self.validation, (Holdout, XXCrossValidation)):
            validation_type = self.validation.__class__.__name__
            results['Validation Type'] = [validation_type]

            if isinstance(self.validation, Holdout):
                train_set, test_set = self.validation.split(pd.concat([self.X, self.y], axis=1))
                self.evaluate_and_store_results(train_set, test_set, results, validation_type)
            elif isinstance(self.validation, XXCrossValidation):
                for i, (train_set, test_set) in enumerate(self.validation.split(pd.concat([self.X, self.y], axis=1))):
                    self.evaluate_and_store_results(train_set, test_set, results, validation_type, i)

                self.calculate_and_print_mean(results, validation_type)

                results['Validation Type'] = ['Mean']

                # Save results to Excel
                self.save_to_excel(results)

                # Plot delle performance
                self.plot_results(results, validation_type)

    def evaluate_and_store_results(self, train_set, test_set, results, validation_type, fold_number=None):
        accuracy, error_rate, specificity, g_mean, sensitivity = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])

        results['Accuracy'].append(accuracy)
        results['Error Rate'].append(error_rate)
        results['Specificity'].append(specificity)
        results['Sensitivity'].append(sensitivity)
        results['Geometric Mean'].append(g_mean)

        if fold_number is not None:
            print(f"{validation_type} {fold_number + 1} Accuracy: {accuracy:.4f}")
        else:
            print(f"{validation_type} Accuracy: {accuracy:.4f}")

    def calculate_and_print_mean(self, results, validation_type):
        mean_accuracy = np.mean(results['Accuracy'])
        mean_error_rate = np.mean(results['Error Rate'])
        mean_specificity = np.mean(results['Specificity'])
        mean_sensitivity = np.mean(results['Sensitivity'])
        mean_g_mean = np.mean(results['Geometric Mean'])

        print(f"\nMean Evaluation across {validation_type}:")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean Error Rate: {mean_error_rate:.4f}")
        print(f"Mean Specificity: {mean_specificity:.4f}")
        print(f"Mean Sensitivity: {mean_sensitivity:.4f}")
        print(f"Mean Geometric Mean: {mean_g_mean:.4f}")

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

    def plot_results(self, results, validation_type):
        # Plot delle performance
        metrics = ['Accuracy', 'Error Rate', 'Specificity', 'Sensitivity', 'Geometric Mean']
        plt.bar(metrics, results['Accuracy'])  # Modifica qui per la metrica che vuoi visualizzare
        plt.title(f'{validation_type} Evaluation Metrics')
        plt.ylabel('Metric Value')
        plt.savefig(f'output/{validation_type.lower()}_evaluation_plot.png')
        plt.show()