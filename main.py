import sys
import pandas as pd
from evaluation import ModelEvaluation
from fetching import data_preprocessing
import matplotlib.pyplot as plt
 

 #pusha ultima modifica
def main():
    # Load and preprocess data
    if len(sys.argv) > 1:
        X, y = data_preprocessing(file=sys.argv[1])
    else:
        X, y = data_preprocessing()
 
# Choose validation type
    validation_type = input("Choose validation type (holdout/xx): ").lower()
    while validation_type not in ['holdout', 'xx']:
       print("Invalid choice. Choose between 'holdout' and 'xx'.")
       validation_type = input("Choose validation type (holdout/xx): ").lower()
 
 
    # Initialize Model Evaluator
    method = validation_type=="holdout"
    evaluator = ModelEvaluation(X, y,method)
 
    # Choose the number of neighbors or folds based on validation type
    if validation_type == 'holdout':
        k_neighbors = int(input("Enter the number of neighbors (k): "))
        test_size=float(input("Enter the test size:"))
        print(X.empty)
        X_train, y_train, X_test, y_test = evaluator.split(X, y, test_size)

    else:  # 'xx' or cross-validation
        k = int(input("Enter the n for experiments "))
        X_train, y_train, X_test, y_test = evaluator.split(X, y,k, random_state=None)



# Evaluate
    choice = select_metrics()
    df_evaluation = evaluator.evaluate(X_train, y_train, X_test, y_test, choice)
    df_evaluation_mean = evaluator.meanCalc(df_evaluation)

 
    # Print the metrics
    evaluator.printMetrics(df_evaluation)
    evaluator.save_metrics(df_evaluation)
    evaluator.metrics_plot(df_evaluation)
 
def select_metrics():
    metrics_mapping = {
        '1': 'Accuracy Rate',
        '2': 'Error Rate',
        '3': 'Sensibility',
        '4': 'Specificity',
        '5': 'Geometric Mean',
        '6':'All the bove'
    }

 
    print("Choose metrics to validate:")
    for key, value in metrics_mapping.items():
        print(f"{key}. {value}")
    metric_choice = input("Enter the number corresponding to the desired option (e.g., '1' for Accuracy Rate): ")
 
    if metric_choice not in metrics_mapping:
        print("Invalid choice. Exiting program.")
        sys.exit()
 
    return metrics_mapping[metric_choice]
 
if __name__ == "__main__":
    main()
