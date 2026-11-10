import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_total_charges(df:pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)

    return df_clean





"""
    Sépare X et y, et fait le split Train/Test (stratifié).
"""

def prepare_data_for_modeling(df: pd.DataFrame, target_col: str = 'Churn'):
    df_model = df.copy()
    df_model[target_col] = df_model[target_col].map({'Yes': 1, 'No': 0})
    
    num_features = ['tenure', 'MonthlyCharges']
    cat_features = [
        'Contract', 'InternetService', 'TechSupport', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'PaymentMethod', 
        'SeniorCitizen', 'Partner', 'PaperlessBilling'
    ]
    
    X = df_model[num_features + cat_features]
    y = df_model[target_col]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    


"""
    prprocessing : Encodage des variables
"""
def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """
    Construit le pipeline de transformation (Standardisation + One-Hot Encoding).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_features)
        ])
    return preprocessor