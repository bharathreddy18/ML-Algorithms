from zenml import step
import pandas as pd
from typing import Tuple
from src.data_splitter import DataSplitter, TrainTestSplit

@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(TrainTestSplit(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test
