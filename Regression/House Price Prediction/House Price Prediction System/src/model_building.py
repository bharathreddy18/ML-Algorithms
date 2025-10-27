import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Abstract Base class for model building.
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        # Returns trained scikit learn model instance
        pass

# Concrete strategy for linear regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError('X_train must be pandas Dataframe.')
        if not isinstance(y_train, pd.Series):
            raise TypeError('y_train must be pandas Series.')
        
        # Creating a pipeline with Standard Scaling and Linear Regression.
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),   # Feature Scaling for all features.
                ('model', LinearRegression())   # Linear Regression model initialization.
            ]
        )

        logging.info('Training Linear Regression Model.')
        pipeline.fit(X_train, y_train)

        logging.info('Model training completed.')
        return pipeline
    
# Context class for model building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info('switching model building strategy.')
        self.strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info('Building and Training the model using the sklearn model.')
        return self.strategy.build_and_train_model(X_train, y_train)
    

    '''
    Create more concrete strategies using different ML models.
    '''