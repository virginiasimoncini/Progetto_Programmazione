# Data preprocessing
This branch process the dataset.Its steps are:
### Load the dataset:
We import the dataset from a file called `breast_cancer.csv`
### Divide dataset in features and targets
We pop the "Class" column from the dataframe and assign it to the Y variable. Then we assign the rest of the dataframe to the X variable.
### Process missing values appropiately:
We must take into account the missing values in the features dataset. For this we have used the "blackfill" method.