# Classe ModelEvaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier  # Assicurati di avere l'import corretto per il tuo algoritmo KNN
import seaborn as sns

class ModelEvaluation:
    def __init__(self,x,y,method):
        self.x = x
        self.y = y
        self.method = method

#Split del dataset in nei set di training e testing:
#o In Holdout, seguendo le percentuali specificate.
    def split(self, x, y, test_size, random_state=None):
        x_train = pd.DataFrame()
        y_train = pd.Series()
        x_test = pd.DataFrame()
        y_test = pd.Series()
        splits = []

        if self.method == "holdout":
            # Assicurati che random_state sia impostato per riproducibilità se specificato
            np.random.seed(random_state)
            
            # Calcola l'indice per il set di test
            indices = x.index.tolist()
            test_indices = np.random.choice(indices, size=int(len(x) * test_size), replace=False)
            
            # Definisci set di training e test basandoti sugli indici
            x_test = x.loc[test_indices]
            y_test = y.loc[test_indices]
            x_train = x.drop(test_indices)
            y_train = y.drop(test_indices)
            
            return x_train, y_train, x_test, y_test

        elif self.method == "xx":
            # Implementazione diretta della LOOCV
            for i in range(len(x)):
                x_train = x.drop(i)
                y_train = y.drop(i)
                x_test = x.iloc[[i]]
                y_test = y.iloc[[i]]
                splits.append((x_train, y_train, x_test, y_test))
            return splits
    
    
    def confMatrixCreation(self, y_test, y_pred):
        true_negative = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        for y, predict in zip(y_test, y_pred):
            if y == predict and predict == 2:
                true_negative += 1
            elif y == predict and predict == 4:
                true_positive += 1
            elif y != predict and predict == 4:
                false_positive += 1
            elif y != predict and predict == 2:
                false_negative += 1
                
        return true_negative, true_positive, false_positive, false_negative
    
    def evaluate(self, x_train, y_train, x_test, y_test, k_neighbors):
        # Train the model on the entire training set
        model = KNNClassifier(k_neighbors)
        model.fit(x_train, y_train)
        
        # Make predictions on the entire test set
        y_pred = model.predict(x_test)
        
        # Calculate confusion matrix elements
        tn, tp, fp, fn = self.confMatrixCreation(y_test, y_pred)
        
        # Calculate accuracy
        accuracy = self.__accuracy__(tp, tn, fp, fn)
        
        # Calculate the metrics based on confusion matrix values and accuracy
        metrics = {
            'accuracy': accuracy,
            'error_rate': self.__error_rate__(accuracy),  # Passa solo accuracy a __error_rate__
            'sensitivity': self.__sensitivity__(tp, fn),
            'specificity': self.__specificity__(tn, fp),
            'geometric_mean': self.__geometric_mean__(tp, tn, fn, fp)    
        }
        
        pd_evaluation = pd.DataFrame([metrics])
        return pd_evaluation


    def __accuracy__(self, true_positive, true_negative, false_positive, false_negative):
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        return accuracy
 
    def __error_rate__(self, accuracy):
        error_rate = 1 - accuracy
        return error_rate
 
    def __sensitivity__(self, true_positive, false_negative):
        # Gestisce il caso di divisione per zero
        if true_positive + false_negative == 0:
            return 0  # Restituisce 0 se non ci sono veri positivi e falsi negativi
        sensitivity = true_positive / (true_positive + false_negative)
        return sensitivity
    
    def __specificity__(self, true_negative, false_positive):
        # Gestisce il caso di divisione per zero
        if true_negative + false_positive == 0:
            return 0  # Restituisce 0 se non ci sono veri negativi e falsi positivi
        specificity = true_negative / (true_negative + false_positive)
        return specificity

    def __geometric_mean__(self, true_positive, true_negative, false_negative, false_positive):
        # Calcola il primo termine, controllando per divisione per zero
        if true_positive + false_negative == 0:
            sensitivity_term = 0  # o np.nan, a seconda della preferenza
        else:
            sensitivity_term = true_positive / (true_positive + false_negative)      
        # Calcola il secondo termine, controllando per divisione per zero
        if true_negative + false_positive == 0:
            specificity_term = 0  # o np.nan, a seconda della preferenza
        else:
            specificity_term = true_negative / (true_negative + false_positive)   
        # Se uno dei termini è np.nan, la media geometrica sarà np.nan
        geometric_mean = np.sqrt(sensitivity_term *specificity_term)
        return geometric_mean

    
    # calcolare la media della metrica attraverso gli
    # esperimenti.
    def meanCalc(self, df_evaluation):
        return df_evaluation.mean()
 
    # • Utilizzare le metriche di validazione speciCicate.
    # • Una volta Finita la validazione del modello. Salva le performance su un File (es. excel) e con un
    # plot 
    
    def printMetrics(self, df, selected_metric):
        # Dizionario che mappa le scelte numeriche delle metriche ai loro nomi nelle colonne del DataFrame
        metrics_map = {
            '1': 'accuracy',
            '2': 'error_rate',
            '3': 'sensitivity',
            '4': 'specificity',
            '5': 'geometric_mean'
        }      
        # Controlla se l'utente ha selezionato l'opzione "All the above"
        if selected_metric == '6':
            # Itera su tutte le metriche disponibili nel dizionario
            for key, metric in metrics_map.items():
                # Controlla se la metrica corrente è presente come colonna nel DataFrame
                if metric in df.columns:
                    # Stampa il nome della metrica in maiuscolo seguito dai suoi valori
                    print(f"{metric.upper()}:")
                    print(df[metric].to_string(index=False))
                else:
                    # Se la metrica non è presente nel DataFrame, stampa che non è disponibile
                    print(f"{metric.upper()}: Not available")
        else:
            # Ottiene il nome della metrica corrispondente alla scelta dell'utente dal dizionario
            metric_name = metrics_map.get(selected_metric, None)
            
            # Controlla se il nome della metrica è valido e presente nel DataFrame
            if metric_name and metric_name in df.columns:
                # Stampa il nome della metrica selezionata in maiuscolo e i suoi valori
                print(f"{metric_name.upper()}:")
                print(df[metric_name].to_string(index=False))
            else:
                # Se la scelta della metrica non è valida, stampa un messaggio di selezione non valida
                print("Invalid metric selection.")    
            
            
    def save_metrics(self, df):
        df.to_csv('output/metrics.csv', index=False)
 
    def metrics_plot(self, df):
        # Trasforma il DataFrame per avere metriche come colonne e valori come righe
        df_melted = df.melt(var_name='Metrics', value_name='Value')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metrics', y='Value', data=df_melted)
        plt.xticks(rotation=45)
        plt.title('Model Performance Metrics')
        # Salva il grafico in un file nella directory 'output'
        plt.savefig('output/model_performance_metrics.png')
        plt.show() 
        # È importante chiudere la figura dopo averla salvata per evitare sovrapposizioni con futuri plot
        plt.close()
        
        
