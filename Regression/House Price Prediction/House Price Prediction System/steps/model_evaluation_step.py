import logging
from zenml import step
# from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluation import ModelEvaluator, RegressionModelEvaluationStrategy

@step(enable_cache=False)
def model_evaluation_step(trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError('X_test must be pandas dataframe.')
    if not isinstance(y_test, pd.Series):
        raise TypeError('y_test must be pandas Series.')
    
    logging.info('Applying the same preprocessing to the test data')
    #Apply the preprocessor
    X_test_processed = trained_model.named_steps['preprocessor'].transform(X_test)

    # Initialize the evaluator
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
    
    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(trained_model.named_steps['model'], X_test_processed, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise TypeError('Evaluation metrics must be returned as a dictionary.')
    # mse = evaluation_metrics.get('Mean Squared Error', None)
    return evaluation_metrics




