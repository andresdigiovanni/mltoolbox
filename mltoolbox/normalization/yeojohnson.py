import pandas as pd
from scipy.stats import yeojohnson


def fit_yeojohnson(data: pd.DataFrame, skewness_threshold: float = 0.5) -> dict:
    lambdas = {}

    for col in data.columns:
        skewness = data[col].skew()

        if skewness_threshold < abs(skewness):
            try:
                _, lambda_value = yeojohnson(data[col])
                lambdas[col] = lambda_value
            except Exception as err:
                print(f"Preprocess error. Column: {col}. Skewness: {skewness}. {err}")

    return lambdas


def transform_yeojohnson(data: pd.DataFrame, lambdas: dict) -> pd.DataFrame:
    transformed_data = data.copy()

    for col, lambda_value in lambdas.items():
        transformed_data[col] = yeojohnson(data[col], lambda_value)

    return transformed_data
