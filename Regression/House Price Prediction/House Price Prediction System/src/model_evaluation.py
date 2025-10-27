import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Abstract Base class for model evaluation strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        pass

# Concrete strategy for Regression model evaluation.
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info('Predicting using the trained model.')
        y_pred = model.predict(X_test)

        logging.info('Calculating the model metrics.')
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

        metrics = {'Mean Squared Error': mse,'R-Squared': r2}
        logging.info(f'Model Evaluation Metrics: {metrics}')
        return metrics
    
# Context class for Model evaluation.
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info('Switching the model evaluation strategy')
        self.strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info('Evaluationg the model using selected strategy.')
        return self.strategy.evaluate_model(model, X_test, y_test)
