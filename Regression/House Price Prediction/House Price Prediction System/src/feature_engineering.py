import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# Setup Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete Strategy for Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df.shape)
        logging.info(f'Applying log transformation to features: {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info('Log transformation completed.')
        print(df_transformed.shape)
        return df_transformed
    
# Concrete strategy for Standard Scaler
class StandardScalerTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Applying Standard Scaling to features: {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = self.scaler.fit_transform(df[feature])
        logging.info('Standard Scaler Transformation Completed.')
        return df_transformed
    
# Concrete strategy for MinMax Scaler
class MinMaxScalerTransformation(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0,1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Applying MinMax Scaling to features: {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = self.scaler.fit_transform(df[feature])
        logging.info('MinMax Scaler Transformation Completed.')
        return df_transformed
    
# Concrete Strategy for One Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f'Applying One Hot Encoding to features: {self.features}')
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info('One Hot encoding completed.')
        return df_transformed
    
# Context class for feature engineering
class FeatureEngineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info(f'Switching Strategy to {self.strategy}')
        self.strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('Applying Feature Engineering Strategy')
        return self.strategy.apply_transformation(df)
    
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\Admin\Desktop\ML Projects\House Price Prediction System\extracted_data\AmesHousing.csv')
    eng = FeatureEngineering(LogTransformation(['Gr Liv Area', 'SalePrice']))
    eng.apply_feature_engineering(df)