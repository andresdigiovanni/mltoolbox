import numpy as np
import pandas as pd


def cyclical_transformation(df: pd.DataFrame, column: str) -> tuple:
    data = df[column]

    return (
        np.sin(data * 2.0 * np.pi / data.max()),
        np.cos(data * 2.0 * np.pi / data.max()),
    )
