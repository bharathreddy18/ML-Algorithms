from zenml import step
import pandas as pd
from src.feature_engineering import FeatureEngineering, LogTransformation, StandardScalerTransformation, MinMaxScalerTransformation, OneHotEncoding

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = 'log', features: list = None) -> pd.DataFrame:
    if strategy == 'log':
        engineer = FeatureEngineering(LogTransformation(features))
    elif strategy == 'standard':
        engineer = FeatureEngineering(StandardScalerTransformation(features))
    elif strategy == 'minmax':
        engineer = FeatureEngineering(MinMaxScalerTransformation(features))
    elif strategy == 'onehot':
        engineer = FeatureEngineering(OneHotEncoding(features))
    else:
        raise ValueError(f'Unsupported Missing Values Strategy: {strategy}')
    engineered_df = engineer.apply_feature_engineering(df)
    return engineered_df