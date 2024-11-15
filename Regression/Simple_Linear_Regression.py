import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
import os

class SLR:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.X = self.df.iloc[:, [1]]
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
            print(f'Testing Accuracy: {self.reg.score(self.X_test, self.y_test)}')

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')

    def plots(self):
        try:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
            ax1.scatter(self.X_train, self.y_train, color='red', marker='*')
            ax1.plot(self.X_train, self.reg.predict(self.X_train), color='b', marker='+')
            ax1.set_xlabel('X_Train')
            ax1.set_ylabel('y_train')
            ax1.set_title("Training Data")

            ax2.scatter(self.X_test, self.y_test, color='red', marker='*')
            ax2.plot(self.X_test, self.reg.predict(self.X_test), color='b', marker='+')
            ax2.set_xlabel('X_Test')
            ax2.set_ylabel('y_test')
            ax2.set_title("Testing Data")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'The Error Type: {exc_type}')
            print(f'The Error Msg: {exc_value}')
            print(f'The Error Line no: {exc_traceback.tb_lineno}')

if __name__ == '__main__':
    cwd = os.getcwd()
    simple = SLR(cwd+'\Salary_dataset.csv')
    simple.training()
    simple.testing_data()
    simple.plots()








