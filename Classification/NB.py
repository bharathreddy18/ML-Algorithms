import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

class Naive_Bayes:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1)
            self.df['diagnosis'] = self.df['diagnosis'].map({'B':0, 'M':1}).astype('int')
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}, Error Value: {er_val}, Error Line: {er_tb.tb_lineno}')

    def training(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.nb = GaussianNB()
            self.nb.fit(self.X_train, self.y_train)
            print('Training Performance')
            print(f'Accuracy score: {self.nb.score(self.X_train, self.y_train)}\n')
            print(f'MSE: {mean_squared_error(self.y_train, self.nb.predict(self.X_train))}\n')
            print(f'Confusion Matrix: {confusion_matrix(self.y_train, self.nb.predict(self.X_train))}\n')
            print(f'Classification Report: {classification_report(self.y_train, self.nb.predict(self.X_train))}\n')
            print('-----------------------------------------------------------------------------------------')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}, Error Value: {er_val}, Error Line: {er_tb.tb_lineno}')

    def testing(self):
        try:
            self.y_test_pred = self.nb.predict(self.X_test)
            print('Testing Performance')
            print(f'Accuracy score: {self.nb.score(self.X_test, self.y_test)}\n')
            print(f'MSE: {mean_squared_error(self.y_test, self.y_test_pred)}\n')
            print(f'Confusion Matrix: {confusion_matrix(self.y_test, self.y_test_pred)}\n')
            print(f'Classification Report: {classification_report(self.y_test, self.y_test_pred)}\n')
            print('-----------------------------------------------------------------------------------------')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}, Error Value: {er_val}, Error Line: {er_tb.tb_lineno}')



if __name__ == "__main__":
    nb = Naive_Bayes('breast_cancer.csv')
    nb.training()
    nb.testing()
