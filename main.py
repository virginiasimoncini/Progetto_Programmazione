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
    validation_type = input("Choose validation type (holdout/xx): ").lower()
    while validation_type not in ['holdout', 'xx']:
        print("Invalid choice. Choose between 'holdout' and 'xx'.")
        validation_type = input("Choose validation type (holdout/xx): ")

    # Choose metrics to validate
    metrics_to_validate = select_metrics()

    # Initialize Model Evaluator
    evaluator = ModelEvaluation(X, y, k_neighbors, validation_type, params={}, chosen_metrics=metrics_to_validate)

    # Perform validation and get results
    evaluator.evaluate_model()

    accuracy, error_rate, sensibility, specificity, geometric_mean = evaluator.get_metrics()

    # Print the metrics
    print_metrics(accuracy, error_rate, sensibility, specificity, geometric_mean)


def select_metrics():
    print("Choose metrics to validate:")
    print("1. Accuracy Rate")
    print("2. Error Rate")
    print("3. Sensibility")
    print("4. Specificity")
    print("5. Geometric Mean")
    
    metric_choice = input("Enter the number corresponding to the desired option (e.g., '1' for Accuracy Rate): ")

    metrics_mapping = {
        '1': 'Accuracy Rate',
        '2': 'Error Rate',
        '3': 'Sensibility',
        '4': 'Specificity',
        '5': 'Geometric Mean',
        'accuracy': 'Accuracy Rate',
        'error rate': 'Error Rate',
        'sensibility': 'Sensitivity',
        'specificity': 'Specificity',
        'geometric mean': 'Geometric Mean',
    }  

    if metric_choice in metrics_mapping:
        metrics_to_validate = [metrics_mapping[metric_choice]]
    else:
        print("Invalid choice. Exiting program.")
        sys.exit()

    return metrics_to_validate

def print_metrics(accuracy_rate, error_rate, sensitivity, specificity, geometric_mean):
    print("Metrics:")
    print(f"Accuracy Rate: {accuracy_rate}")
    print(f"Error Rate: {error_rate}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Geometric Mean: {geometric_mean}")

if __name__ == "__main__":
    main()
