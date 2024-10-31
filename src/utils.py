import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# For each column, convert nulls to np.nan, and convert to float if possible
def convert_cols_to_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({
                "": np.nan,
                "nan": np.nan,
                "NAN": np.nan,
                "NaN": np.nan,
                "na": np.nan
                })
            if df[col].str.isnumeric().all():
                df[col] = df[col].astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def custom_scorer(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    true_positives = conf_matrix[1, 1]
    false_positives = conf_matrix[0, 1]
    true_negatives = conf_matrix[0, 0]
    false_negatives = conf_matrix[1, 0]
    # Calculate cost of the model
    inspection_cost_min = 100 * (true_positives + false_positives)
    inspection_cost_max = 300 * (true_positives + false_positives)
    voucher_cost = 250 * false_positives
    repair_cost = 150 * true_positives
    failed_repair_cost_min = 750 * false_negatives
    failed_repair_cost_max = 2000 * false_negatives
    total_cost_min = inspection_cost_min + voucher_cost + repair_cost + failed_repair_cost_min
    total_cost_max = inspection_cost_max + voucher_cost + repair_cost + failed_repair_cost_max
    total_cost_avg = (total_cost_min + total_cost_max) / 2
    return total_cost_avg
