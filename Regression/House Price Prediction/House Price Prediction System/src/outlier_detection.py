import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base class for Outlier Detection
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete Strategy for Z-Score based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting Outliers using Z-Score method')
        numeric_cols = df.select_dtypes(include=np.number).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        outliers = z_scores > self.threshold
        logging.info(f'Outliers detected with threshold: {self.threshold}')
        return outliers
    
# Concrete Strategy for IQR based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Detecting Outliers using IQR method')
        numeric_cols = df.select_dtypes(include=np.number).columns
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols ] > (Q3 + 1.5 * IQR))
        logging.info('Outliers are detected using IQR method')
        return outliers
    
# Context class for Outlier detection and handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info('switching outlier detection strategy')
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Executing outlier detection strategy.')
        return self.strategy.detect_outliers(df)
    
    def handle_outliers(self, df: pd.DataFrame, method='remove', **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        
        if method == 'remove':
            logging.info('Removing outliers from the dataset.')
            df_cleaned = df[~outliers.any(axis=1)]
        
        elif method == 'cap':
            logging.info('Capping outliers in the dataset.')
            numeric_cols = df.select_dtypes(include=np.number).columns
            Q1 = df[numeric_cols].quantile(0.01)
            Q3 = df[numeric_cols].quantile(0.99)
            df_cleaned = df.copy()
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].clip(lower=Q1, upper=Q3)
        
        else:
            logging.warning(f'Unknown method: {method}. No outlier handling performed.')
            return df
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            if feature not in df.columns:
                logging.warning(f'Feature {feature} not found in dataframe. Skipping')
                continue
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\Admin\Desktop\ML Projects\House Price Prediction System\extracted_data\AmesHousing.csv')
    print(df.shape)
    outlier = OutlierDetector(ZScoreOutlierDetection(3))
    cleaned_df = outlier.handle_outliers(df, method='remove')
    print(cleaned_df.shape)

