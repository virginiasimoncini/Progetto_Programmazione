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
        train_indices = shuffled_indices[self.test_size:]

        train_set = df.iloc[train_indices]
        test_set = df.iloc[test_indices]

        return train_set, test_set

class XXCrossValidation:
    def __init__(self, k):
        self.k = k

    def split(self, df):
        folds = []

        np.random.seed(42)
        indices = np.random.permutation(df.index)
        features_mixed = df.loc[indices]

        fold_length = len(df) // self.k

        for i in range(self.k):
            test_start, test_end = i * fold_length, (i + 1) * fold_length 
            test_set = features_mixed.iloc[test_start:test_end]
            train_set = pd.concat([features_mixed.iloc[:test_start], features_mixed.iloc[test_end:]])
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

    def evaluate_validation(self):
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
            results_df.to_excel('validation_results.xlsx', index=False)

            # Plot delle performance
            plt.bar(['Accuracy', 'Error Rate', 'Specificity', 'Geometric Mean'], [accuracy, error_rate, specificity, g_mean])
            plt.title('Holdout Evaluation Metrics')
            plt.ylabel('Metric Value')
            plt.savefig('holdout_evaluation_plot.png')
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
            results_df.to_excel('validation_results.xlsx', index=False)

            # Plot delle performance
            for metric in ['Accuracy', 'Error Rate', 'Specificity', 'Geometric Mean']:
                plt.plot(results_df['Validation Type'], results_df[metric], label=metric)

            plt.title('Cross-Validation Evaluation Metrics')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.savefig('crossval_evaluation_plot.png')
            plt.show()
