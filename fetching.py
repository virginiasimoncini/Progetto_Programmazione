import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
import numpy as np


def data_preprocessing():
    # fetch dataset 
    df = pd.read_csv("breast_cancer.csv") 
  
    # data (as pandas dataframes)
    y = df.pop("Class")
    x = df
    # normalize features
    for column in x.columns: 
        x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())       

    # Fixing missing values appropiately (We only analyse the features given that every instance must have a target):
    x.bfill()
    print(x.head())
    print(y.head())
    #return the targets and features
    return x,y