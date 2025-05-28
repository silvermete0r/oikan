import pandas as pd

data = pd.read_csv('benchmarking/PMLB - datasets/classification_benchmark_results.csv')

def format_data(data):
    columns_to_format = ['train_time', 'train_memory', 'predict_time', 'predict_memory', 'accuracy_oikan', 'precision_oikan', 'f1_oikan', 'accuracy_elasticnet', 'precision_elasticnet', 'f1_elasticnet']
    for column in columns_to_format:
        if column in data.columns:
            data[column] = data[column].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else 'N/A')
    return data

formatted_data = format_data(data)
formatted_data.to_csv('benchmarking/PMLB - datasets/formatted_classification_benchmark_results.csv', index=False)