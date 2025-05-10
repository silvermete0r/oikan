import numpy as np
import torch
import time
import tracemalloc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pmlb import fetch_data
from oikan import OIKANRegressor, OIKANClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def adjust_augmentation_factor(n_samples, augmentation_factor=10, max_augmented_size=50000):
    """Adjust augmentation_factor to ensure augmented data size <= max_augmented_size."""
    if augmentation_factor * n_samples > max_augmented_size:
        # Reduce augmentation_factor to keep total augmented samples <= 50000
        augmentation_factor = max(1, max_augmented_size // n_samples)
    return augmentation_factor

def benchmark_regression_dataset(dataset_name):
    """Benchmark OIKANRegressor on a regression dataset."""
    # Fetch data
    data = fetch_data(dataset_name)
    X = data.drop('target', axis=1).values
    y = data['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Adjust augmentation factor based on training data size
    augmentation_factor = adjust_augmentation_factor(len(X_train))
    
    # Initialize model with adjusted augmentation factor
    model = OIKANRegressor(augmentation_factor=augmentation_factor, verbose=True)
    
    # Training benchmark
    tracemalloc.start()
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    _, train_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Prediction benchmark
    tracemalloc.start()
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    _, predict_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'dataset': dataset_name,
        'train_time': train_time,
        'train_memory': train_memory / 10**6,  # Convert to MB
        'predict_time': predict_time,
        'predict_memory': predict_memory / 10**6,  # Convert to MB
        'MSE': mse,
        'R2': r2,
        'augmentation_factor': augmentation_factor
    }

def benchmark_classification_dataset(dataset_name):
    """Benchmark OIKANClassifier on a classification dataset."""
    # Fetch data
    data = fetch_data(dataset_name)
    X = data.drop('target', axis=1).values
    y = data['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Adjust augmentation factor based on training data size
    augmentation_factor = adjust_augmentation_factor(len(X_train))
    
    # Initialize model with adjusted augmentation factor
    model = OIKANClassifier(augmentation_factor=augmentation_factor, verbose=True)
    
    # Training benchmark
    tracemalloc.start()
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    _, train_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Prediction benchmark
    tracemalloc.start()
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    _, predict_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'dataset': dataset_name,
        'train_time': train_time,
        'train_memory': train_memory / 10**6,  # Convert to MB
        'predict_time': predict_time,
        'predict_memory': predict_memory / 10**6,  # Convert to MB
        'accuracy': accuracy,
        'augmentation_factor': augmentation_factor
    }

if __name__ == "__main__":
    print("Starting OIKAN Advanced Auto-Benchmarking...\n")
    
    # Confirmed regression datasets
    regression_datasets = [
        '1027_ESL',
        '1028_SWD',
        '1029_LEV',
        '1030_ERA',
        '1089_USCrime',
        '1096_FacultySalaries',
        '228_elusage',
        '229_pwLinear',
        '230_machine_cpu',
        '485_analcatdata_vehicle'
    ]
    
    # Confirmed classification datasets
    classification_datasets = [
        'credit_g',
        'mushroom',
        'iris',
        'wine_recognition',
        'breast_cancer_wisconsin',
        'tic_tac_toe',
        'analcatdata_authorship',
        'analcatdata_fraud',
        'penguins',
        'monk3'
    ]
    
    # Benchmark regression datasets
    regression_results = []
    for dataset in regression_datasets:
        try:
            result = benchmark_regression_dataset(dataset)
            regression_results.append(result)
            print(f"Completed regression benchmark for {dataset} (augmentation_factor={result['augmentation_factor']})")
        except Exception as e:
            print(f"Error benchmarking regression dataset {dataset}: {e}")
    regression_df = pd.DataFrame(regression_results)
    
    # Benchmark classification datasets
    classification_results = []
    for dataset in classification_datasets:
        try:
            result = benchmark_classification_dataset(dataset)
            classification_results.append(result)
            print(f"Completed classification benchmark for {dataset} (augmentation_factor={result['augmentation_factor']})")
        except Exception as e:
            print(f"Error benchmarking classification dataset {dataset}: {e}")
    classification_df = pd.DataFrame(classification_results)
    
    # Display results
    print("\nRegression Benchmark Results:")
    print(regression_df.to_string(index=False))
    print("\nClassification Benchmark Results:")
    print(classification_df.to_string(index=False))
    
    # Save results to CSV files
    regression_df.to_csv('regression_benchmark_results.csv', index=False)
    classification_df.to_csv('classification_benchmark_results.csv', index=False)
    
    print("\nBenchmarking completed. Results saved to 'regression_benchmark_results.csv' and 'classification_benchmark_results.csv'.")