import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

class MLR:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.df['State'] = self.df['State'].map({'New York': 0, 'California':1, 'Florida':2}).astype(int)
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')

    def training(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.reg = LinearRegression()
            self.reg.fit(self.X_train, self.y_train)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')

    def testing_data(self):
        try:
            print(f'Training Accuracy: {self.reg.score(self.X_train, self.y_train)}')
            print(f'Training Loss: {mean_squared_error(self.y_train, self.reg.predict(self.X_train))}')
            print(f'Testing Accuracy: {self.reg.score(self.X_test, self.y_test)}')
            print(f'Training Loss: {mean_squared_error(self.y_test, self.reg.predict(self.X_test))}')

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')

    def r2_score(self):
        try:
            self.y_train = np.array([self.y_train])
            self.y_train_pred = np.array([self.reg.predict(self.X_train)])
            self.y_mean = np.mean(self.y_train)
            self.SSE = np.sum((self.y_train - self.y_train_pred) ** 2)
            self.SST = np.sum((self.y_train - self.y_mean) ** 2)

            self.r2_score = 1 - (self.SSE / self.SST)
            print(f'Training R2-score: {self.r2_score}')

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')


if __name__ == '__main__':
    cwd = os.getcwd()
    multiple = MLR(cwd+'/50_Startups.csv')
    multiple.training()
    multiple.testing_data()
    multiple.r2_score()









