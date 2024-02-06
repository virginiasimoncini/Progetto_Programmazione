import numpy as np
from operator import itemgetter

class KNNClassifier:
    def __init__(self, k):
        # Definizione del default constructor, k è il numero di vicini da considerare
        self.k = k

    def fit(self, X, y):
        # Memorizza i dati di addestramento
        self.X_train = X
        self.y_train = y

    # self.X_train e self.y_train memorizzano i dati di addestramento all'interno dell'istanza della classe. 
    # Dopo aver chiamato fit sulla tua istanza della classe, accedo ai dati di addestramento tramite self.X_train e self.y_train.

    def euclidean_distance(self, x1, x2):
        # Calcola la distanza euclidea tra due vettori
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def model_prediction(self, X):
        y_predictions = [] # Lista dove verranno salvate le previsioni
        for _, test_point in X.iterrows(): 
            distances = [] # Elenco delle distanze tra il punto x_test e i punti X
            for index, train_point in self.X_train.iterrows():
                # Calcola le distanze tra il dato di x_test e i dati di X
                dist = self.euclidean_distance(train_point, test_point)
                distances.append((dist, self.y_train[index]))

            # Ordina le distanze in ordine crescente
            distances = sorted(distances, key=itemgetter(0), reverse=False)

            # Seleziona le prime k distanze
            k_nearest_neighbors = distances[:self.k]

            # Estrae le classi corrispondenti ai k vicini più vicini
            k_neighbor_classes = [neighbor[1] for neighbor in k_nearest_neighbors]

            # Supponendo che 'k_neighbors' sia la lista delle classi corrispondenti ai k vicini più prossimi
            unique_elements, counts_elements = np.unique(k_neighbor_classes, return_counts=True)

            # Verifica se c'è solo una classe tra i k vicini più prossimi
            # o se c'è un pareggio tra le occorrenze delle classi
            if len(unique_elements) == 1 or np.max(counts_elements) == np.min(counts_elements):
                # Se c'è solo una classe o c'è un pareggio, scegli in modo casuale
                random_choice = np.random.choice(unique_elements)
                # Aggiungi la scelta casuale alle previsioni
                y_predictions.append(random_choice)
            else:
                # Altrimenti, scegli la classe più frequente
                max_index = np.argmax(counts_elements) 
                # Trova l'indice della classe con il numero massimo di occorrenze tra i k vicini più prossimi
                y_predictions.append(unique_elements[max_index]) 
                # Aggiunge la classe corrispondente all'indice trovato alle previsioni

        return np.array(y_predictions) # questa funzione restituisce le previsioni convertite in un array NumPy





       


        

            
   
