from ucimlrepo import fetch_ucirepo 
import pandas as pd

def data_preprocessing():
    # fetch dataset 
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
    # data (as pandas dataframes) 
    x = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets
    # normalize features
    for column in x.columns: 
        x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())       

    # Fixing missing values appropiately (We only analyse the features given that every instance must have a target):
    x.fillna(method='bfill')

    #return the targets and features
    return x,y