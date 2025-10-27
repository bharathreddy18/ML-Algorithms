import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

# Setup Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base class for Missing Value Handling Strategy.
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df.shape)
        logging.info(f'Dropping missing values with axis={self.axis} and threshold={self.thresh}')
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info('Missing Values Dropped.')
        print(df_cleaned.shape)
        return df_cleaned

# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method = 'mean', fill_value = None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Filling Missing values using method: {self.method} and fill value: {self.fill_value}')
        
        df_cleaned = df.copy()
        if self.method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include='number').columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include='number').columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        elif self.method == 'mode':
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].fillna(df[column].mode()[0])
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f'Unknown method: {self.method}, No Missing Values Handles')
        logging.info('Missing Values Filled.')
        return df_cleaned

# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info('switching missing value strategy.')
        self._strategy = strategy
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Handling missing value strategy.')
        return self._strategy.handle(df)
    


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Admin\Desktop\ML Projects\House Price Prediction System\extracted_data\AmesHousing.csv')
    handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=int(0.9*df.shape[1])))
    handler.handle_missing_values(df)


        