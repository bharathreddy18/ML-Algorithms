import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from zenml import Model, pipeline           
from steps.data_ingestion_step import data_ingestion_step
from steps.handling_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step


@pipeline(
    model = Model(
        # The name uniquely identifies this model.
        name='prices_predictor'
    )
)

def ml_pipeline():
    # Define an end-to-end machine learning pipeline.
    # Data Ingestion step
    raw_data = data_ingestion_step('C:/Users/Admin/Desktop/ML Projects/House Price Prediction System/archive.zip')

    # Handling Missing Values
    filled_data = handle_missing_values_step(raw_data)
    # print("After Handling Missing Values:", filled_data.columns)
    
    # Feature Engineering step
    engineered_data = feature_engineering_step(filled_data, strategy='log', features=['Gr Liv Area', 'SalePrice'])
    # print("After Feature Engineering:", engineered_data.columns)
    
    # Outlier Detection step
    clean_data = outlier_detection_step(engineered_data, strategy='zscore')
    # print("After Outlier Detection:", clean_data.columns)
    
    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column='SalePrice')
    # print("Final Features in X_train:", X_train.columns)

    # Model building step
    model = model_building_step(X_train = X_train, y_train = y_train)

    # Model evaluation
    evaluation_metrics = model_evaluation_step(trained_model = model, X_test=X_test, y_test=y_test)

    return model

if __name__ == "__main__":
    run = ml_pipeline()
