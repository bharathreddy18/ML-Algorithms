import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DT:
    def __init__(self):
        try:
            # Load the dataset
            self.df = pd.read_csv('House_Price_Prediction.csv')
            self.df = self.df[(np.abs(zscore(self.df)) < 3).all(axis=1)]

            # Drop unnecessary columns (if applicable)
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop('Unnamed: 0', axis=1)

            # Features (X) and Target (y)
            self.X = self.df.iloc[:, 1:]  # Assuming features start from 2nd column
            self.y = self.df.iloc[:, 0]  # Assuming the target is in the 1st column
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f"Error type: {er_type}\nError Value: {er_val}\nError Line: {er_tb.tb_lineno}")

    def training(self):
        try:
            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Train the Decision Tree Regressor
            self.dt = DecisionTreeRegressor(random_state=42)
            self.dt.fit(self.X_train, self.y_train)

            # Predict on training data
            self.y_train_pred = self.dt.predict(self.X_train)

            # Training Performance
            print('----------------------TRAINING PERFORMANCE---------------------------')
            print(f'Mean Absolute Error (MAE): {mean_absolute_error(self.y_train, self.y_train_pred):.2f}')
            print(f'Mean Squared Error (MSE): {mean_squared_error(self.y_train, self.y_train_pred):.2f}')
            print(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(self.y_train, self.y_train_pred)):.2f}')
            print(f'R-squared (R²): {r2_score(self.y_train, self.y_train_pred):.2f}')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f"Error type: {er_type}\nError Value: {er_val}\nError Line: {er_tb.tb_lineno}")

    def testing(self):
        try:
            # Predict on test data
            self.y_test_pred = self.dt.predict(self.X_test)

            # Testing Performance
            print('-----------------------TESTING PERFORMANCE---------------------------')
            print(f'Mean Absolute Error (MAE): {mean_absolute_error(self.y_test, self.y_test_pred):.2f}')
            print(f'Mean Squared Error (MSE): {mean_squared_error(self.y_test, self.y_test_pred):.2f}')
            print(f'Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(self.y_test, self.y_test_pred)):.2f}')
            print(f'R-squared (R²): {r2_score(self.y_test, self.y_test_pred):.2f}')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f"Error type: {er_type}\nError Value: {er_val}\nError Line: {er_tb.tb_lineno}")


if __name__ == "__main__":
    try:
        # Initialize and execute the Decision Tree regression
        tree = DT()
        tree.training()
        tree.testing()
    except Exception as e:
        er_type, er_val, er_tb = sys.exc_info()
        print(f"Error type: {er_type}\nError Value: {er_val}\nError Line: {er_tb.tb_lineno}")
