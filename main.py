# Main
import sys
import pandas as pd
from evaluation import ModelEvaluation
from fetching import data_preprocessing
import matplotlib.pyplot as plt

def main():
    # Load and preprocess data
    if len(sys.argv) > 1:
        X, y = data_preprocessing(file=sys.argv[1])
    else:
        X, y = data_preprocessing()

    # Choose the number of neighbors
    k_neighbors = int(input("Enter the number of neighbors (k): "))
    
    # Choose validation type
    validation_type = ""
    while validation_type not in ['holdout', 'xx']:
        print("Invalid choice. Choose between 'holdout' and 'xx'.")
        validation_type = input("Choose validation type (holdout/xx): ".lower())

    # Initialize Model Evaluator
    method = validation_type=="holdout"
    evaluator = ModelEvaluation(X, y,method)
    # Split
    var = select_var()
    x_train,y_train,x_test,y_test = evaluator.split(X,y,var)
    # Evaluate
    choice = select_metrics()
    df_evaluation = evaluator.evaluate(x_train,y_train,x_test,y_test,choice)
    df_mean_evaluation = evaluator.meanCalc(df_evaluation)
    # Print the metrics
    evaluator.printMetrics(df_evaluation)
    evaluator.save_metrics(df_evaluation)
    evaluator.metrics_plot(df_evaluation)


def select_metrics():
    print("Choose metrics to validate:")
    print("1. Accuracy Rate")
    print("2. Error Rate")
    print("3. Sensibility")
    print("4. Specificity")
    print("5. Geometric Mean")
    
    metric_choice = input("Enter the number corresponding to the desired option (e.g., '1' for Accuracy Rate): ")
 

    if not metric_choice in metrics_mapping:
        print("Invalid choice. Exiting program.")
        sys.exit()

    return metrics_to_validate

def select_var():
    print("Choose metrics to validate:")
    print("1. Accuracy Rate")
    print("2. Error Rate")
    print("3. Sensibility")
    print("4. Specificity")
    print("5. Geometric Mean")
    
    metric_choice = input("Enter the number corresponding to the desired option (e.g., '1' for Accuracy Rate): ")
 

    if not metric_choice in metrics_mapping:
        print("Invalid choice. Exiting program.")
        sys.exit()

    return metrics_to_validate

if __name__ == "__main__":
    main()