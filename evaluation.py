# Classe ModelEvaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn import KNNClassifier,predict  # Assicurati di avere l'import corretto per il tuo algoritmo KNN

class ModelEvaluation:
    def __init__(self,x,y,method):
        self.x = x
        self.y = y
        self.holdout = method

#Split del dataset in nei set di training e testing:
#o In Holdout, seguendo le percentuali speciCicate.
    def split(self,x,y,var):
        if(self.holdout):
            #split holdout var = percent_test TODO
            x_train = x.sample(frac = var)
            x_test =  x.sample(frac = (1-var))
            y_train = y.sample(frac = var)
            y_test =  y.sample(frac = (1-var))
        else:
            #split xx var = k TODO
            print ("xx")
        return x_train,y_train,x_test,y_test
<<<<<<< HEAD
    def confMatrixCreation(self):
        true_negative=0
        true_positive=0
        false_positive=0
        false_negative=0

        for y, predict in zip(self.y_test, self.predict):
           if y == predict and predict == 2:
              true_negative += 1
           elif y == predict and predict == 4:
              true_positive += 1
           elif y != predict and predict == 4:
              false_positive += 1
           elif y != predict and predict == 2:
              false_negative += 1

        return true_negative, true_positive, false_positive, false_negative
        #define empty 2x2 matrix
        #take a series of X 
        #calculate predicted_y of x
        #compare and sum to the respective cell
        return null

# valutare le prestazioni del modello sui dati di test
    def evaluate(self,x_train,y_train,x_test,y_test,choice):
        evaluation = []
        #aevaluation is a dataframe and each function should return a column that can be added to the dataframe
        match choice:
            case 1:
                evaluation.insert(self.__accuracy__())
            case 2:
                evaluation.insert(self.__error_rate__())
            case 3:
                evaluation.insert(self.__sensibility__())
            case 4:
                evaluation.insert(self.__specificity__())
            case 5:
                evaluation.insert(self.__geometric_mean__())
            case 6:
                evaluation.insert(self.__accuracy__())
                evaluation.insert(self.__error_rate__())
                evaluation.insert(self.__sensibility__())
                evaluation.insert(self.__specificity__())
                evaluation.insert(self.__geometric_mean__())
        
        pd_evaluation = pd.DataFrame(evaluation)
        return pd_evaluation

    def __accuracy__(self):
        #calculate Accuracy TODO
        return accuracy

    def __error_rate__(self):
        #calculate error rate TODO
        return error_rate

    def __sensibility__(self):
        #TODO
        return sensibility

    def __specificity__(self):
        #TODO
        return specifity

    def __geometric_mean__(self):
        #TODO
        return geometric_mean
# calcolare la media della metrica attraverso gli
#esperimenti.
    def meanCalc(self,df_evaluation):
        return df_mean_evaluation
#• Utilizzare le metriche di validatazione speciCicate.
#• Una volta Finita la validazione del modello. Salvate le performance su un File (es. excel) e con un
#plot (decidete voi quale sia il più adatto).
    def printMetrics(self,df):
        #TODO
        print(df)

=======
    def confMatrixCreation(self,x,y):
        #define empty 2x2 matrix
        #take a series of X 
        #calculate predicted_y of x
        #compare and sum to the respective cell
        return null

# valutare le prestazioni del modello sui dati di test
    def evaluate(self,x_train,y_train,x_test,y_test,choice):
        evaluation = []
        #aevaluation is a dataframe and each function should return a column that can be added to the dataframe
        match choice:
            case 1:
                evaluation.insert(self.__accuracy__())
            case 2:
                evaluation.insert(self.__error_rate__())
            case 3:
                evaluation.insert(self.__sensibility__())
            case 4:
                evaluation.insert(self.__specificity__())
            case 5:
                evaluation.insert(self.__geometric_mean__())
            case 6:
                evaluation.insert(self.__accuracy__())
                evaluation.insert(self.__error_rate__())
                evaluation.insert(self.__sensibility__())
                evaluation.insert(self.__specificity__())
                evaluation.insert(self.__geometric_mean__())
        
        pd_evaluation = pd.DataFrame(evaluation)
        return pd_evaluation

    def __accuracy__(self):
        #calculate Accuracy TODO
        return accuracy

    def __error_rate__(self):
        #calculate error rate TODO
        return error_rate

    def __sensibility__(self):
        #TODO
        return sensibility

    def __specificity__(self):
        #TODO
        return specifity

    def __geometric_mean__(self):
        #TODO
        return geometric_mean
# calcolare la media della metrica attraverso gli
#esperimenti.
    def meanCalc(self,df_evaluation):
        return df_mean_evaluation
#• Utilizzare le metriche di validatazione speciCicate.
#• Una volta Finita la validazione del modello. Salvate le performance su un File (es. excel) e con un
#plot (decidete voi quale sia il più adatto).
    def printMetrics(self,df):
        #TODO
        print(df)

>>>>>>> b078ecdec1fb6a7096fe1cbf7e94135969cd58f8
    def save_metrics(self, df):
        df.to_csv('output/metrics.csv', index=False)
        
    def metrics_plot(self, df):
        df.head()