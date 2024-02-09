import numpy as np
from operator import itemgetter

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_predictions = []  # Lista dove verranno salvate le previsioni
        for _, test_point in X.iterrows():
            distances = []  # Elenco delle distanze tra il punto x_test e i punti X
            for index, train_point in self.X_train.iterrows():
                dist = self.euclidean_distance(train_point, test_point)
                distances.append((dist, self.y_train[index]))

            distances = sorted(distances, key=itemgetter(0), reverse=False)

            k_nearest_neighbors = distances[:self.k]
            k_neighbor_classes = [neighbor[1] for neighbor in k_nearest_neighbors]

            unique_elements, counts_elements = np.unique(k_neighbor_classes, return_counts=True)

            if len(unique_elements) == 1 or np.max(counts_elements) == np.min(counts_elements):
                # Seleziona casualmente se c'è una sola classe o un pareggio
                random_choice = np.random.choice(unique_elements)
                y_predictions.append(random_choice)
            else:
                # Seleziona la classe più frequente
                max_index = np.argmax(counts_elements)
                y_predictions.append(unique_elements[max_index])

        return np.array(y_predictions)
