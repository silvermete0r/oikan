import pandas as pd

data = pd.read_csv('benchmarking/PMLB - datasets/regression_benchmark_results.csv')

def format_data(data):
    columns_to_format = ['train_time', 'train_memory', 'predict_time', 'predict_memory', 'rmse_oikan', 'r2_oikan', 'mape_oikan', 'rmse_elasticnet', 'r2_elasticnet', 'mape_elasticnet']
    for column in columns_to_format:
        if column in data.columns:
            data[column] = data[column].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else 'N/A')
    return data

formatted_data = format_data(data)
formatted_data.to_csv('benchmarking/PMLB - datasets/formatted_regression_benchmark_results.csv', index=False)