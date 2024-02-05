import numpy as np

class KNNClassifier:
    def __init__(self, k, X, y):
        # Inizializza un'istanza del classificatore KNN con il parametro k specificato
        self.k = k
        self.X_train = X
        self.y_train = y   

    def euclidean_distance(self, x1, x2):
        # Calcola la distanza euclidea tra due vettori
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def getKClosestLabels(self, X):
        y_pred = []
        for x in X:
            # Calcola le distanze tra il dato di test e tutti i dati di addestramento
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

            # Trova gli indici dei k vicini più prossimi
            k_indices = np.argsort(distances)[:self.k]

            # Ottieni le etichette corrispondenti agli indici trovati
            k_labels = [self.y_train[i] for i in k_indices]

        return k_labels
    # Supponiamo che k_labels sia un array contenente le etichette (labels) assegnate a qualche insieme di dati.

    # Calcola gli elementi unici e le loro frequenze all'interno dell'array k_labels.
    unique_labels, label_counts = np.unique(k_labels, return_counts=True)

    # Trova la frequenza massima tra tutte le etichette.
    max_count = np.max(label_counts)

    # Crea una lista contenente le etichette più comuni, ovvero quelle con la frequenza massima.
    most_common_labels = [label for label, count in zip(unique_labels, label_counts) if count == max_count]
            
    def predictX(self, X):
        k_labels = self.getKClosestLabels(self, X)
        # Bisogna assicurarsi che k_labels non sia vuoto prima di tentare di trovare l'etichetta più comune
        if not k_labels:
            raise ValueError("L'array k_labels è vuoto. Assicurati di fornire dati validi.")
        # Trova l'etichetta più comune tra i k vicini più prossimi
        most_common_label = max(set(k_labels), key=k_labels.count)
        return most_common_label
