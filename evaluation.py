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
    def __init__(self, X, y, validation, k):
        self.X = X
        self.y = y
        self.validation = validation
        self.k = k

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

        # Calcolo della geometric mean
        recall = np.sum((y_pred == 4) & (y_test == 4)) / np.sum(y_test == 4)
        g_mean = np.sqrt(specificity * recall)

        return accuracy, error_rate, specificity, g_mean


    def evaluate_validation(self, metric_choice):
        if not os.path.exists("output"):
            os.makedirs("output")

        if isinstance(self.validation, Holdout):
            train_set, test_set = self.validation.split(pd.concat([self.X, self.y], axis=1))
            accuracy, error_rate, specificity, g_mean = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])

            print("Holdout Evaluation:")
            if metric_choice == '1':
                print(f"Accuracy: {accuracy:.4f}")
                metric_name = 'Accuracy'
                metric_value = accuracy
            elif metric_choice == '2':
                print(f"Error Rate: {error_rate:.4f}")
                metric_name = 'Error Rate'
                metric_value = error_rate
            elif metric_choice == '3':
                print(f"Specificity: {specificity:.4f}")
                metric_name = 'Specificity'
                metric_value = specificity
            elif metric_choice == '4':
                print(f"Geometric Mean: {g_mean:.4f}")
                metric_name = 'Geometric Mean'
                metric_value = g_mean
            else:
                print("Opzione non valida.")
                return

            # Salvataggio della metrica scelta in Excel
            results_df = pd.DataFrame({
                'Validation Type': ['Holdout'],
                'Metric': [metric_name],
                'Value': [metric_value],
            })
            results_df.to_excel('output/validation_results.xlsx', index=False)

            # Plot della performance
            plt.bar([metric_name], [metric_value])
            plt.title('Holdout Evaluation Metrics')
            plt.ylabel('Metric Value')
            plt.savefig(f'output/{metric_name.lower().replace(" ", "_")}_evaluation_plot.png')
            plt.show()

        elif isinstance(self.validation, XXCrossValidation):
            # La logica per la cross-validation può essere adattata in modo simile
            # Se desideri implementare la stessa scelta di metrica per la cross-validation
            pass

        elif isinstance(self.validation, XXCrossValidation):
            accuracies = []

            validation_type_list = [f'Fold {i + 1}' for i in range(self.validation.num_folds)] if self.validation.num_folds is not None else ['Mean']

            for i, (train_set, test_set) in enumerate(self.validation.split(pd.concat([self.X, self.y], axis=1))):
                accuracy, _, _, _ = self.evaluate(train_set.iloc[:, :-1], test_set.iloc[:, :-1], train_set.iloc[:, -1], test_set.iloc[:, -1])
                accuracies.append(accuracy)

                print(f"Fold {i + 1} Accuracy: {accuracy:.4f}")

            mean_accuracy = np.mean(accuracies)

            print(f"\nMean Accuracy across {self.validation.num_folds} folds: {mean_accuracy:.4f}")

            num_items = len(validation_type_list)
            accuracies += [mean_accuracy] * (num_items - len(accuracies))

            assert len(accuracies) == num_items, "Le liste devono avere la stessa lunghezza"

            results_df = pd.DataFrame({
                'Validation Type': validation_type_list,
                'Accuracy': accuracies
            })
            results_df.to_excel('output/validation_results.xlsx', index=False)

            plt.bar(['Accuracy'], [mean_accuracy])
            plt.title('Cross-Validation Evaluation Metrics')
            plt.ylabel('Mean Accuracy')
            plt.savefig('output/crossval_evaluation_plot.png')
            plt.show()
