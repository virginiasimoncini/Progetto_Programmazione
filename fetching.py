import pandas as pd

def data_preprocessing(file = "breast_cancer.csv"):
    # fetch dataset
    df = pd.read_csv(file) 
  
    # data (as pandas dataframes)
    y = df.pop("Class")
    x = df
    # normalize features
    for column in x.columns: 
        x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())       

    # Fixing missing values appropiately (We only analyse the features given that every instance must have a target):
    x.fillna(df.mean())
    print(x.head())
    print(y.head())
    #return the targets and features
    return x,y