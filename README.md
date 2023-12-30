# Data preprocessing
This branch process the dataset.Its steps are:
### Load the dataset:
By means of the `ucimlrepo` library we download the dataset,
### Divide dataset in features and targets
Due to the type given by `ucimlrepo` we need to separate the dataset now in order to benefit from `panda` 's dataframe functions.
### Process missing values appropiately:
We must take into account the missing values in the features dataset. For this we have used the "blackfill" method.