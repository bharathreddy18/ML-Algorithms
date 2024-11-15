import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix


class KNN:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1)
            self.df['diagnosis'] = self.df['diagnosis'].map({'B':0, 'M':1}).astype('int')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')

    def training_data(self):
        try:
            self.X = self.df.iloc[:,1:]
            self.y = self.df.iloc[:,0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.reg = KNeighborsClassifier()
            self.reg.fit(self.X_train, self.y_train)
            print(f'The Accuracy of Training data: {self.reg.score(self.X_train, self.y_train)}')
            print(f'The MSE of Training data: {mean_squared_error(self.y_train, self.reg.predict(self.X_train))}')
            print(f'The Confusion Matrix: {confusion_matrix(self.y_train, self.reg.predict(self.X_train))}')
            print(f'The Classification report: {classification_report(self.y_train, self.reg.predict(self.X_train))}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')

    def testing_data(self):
        try:
            print(f'The Accuracy of Testing data: {self.reg.score(self.X_test, self.y_test)}')
            print(f'The MSE of Testing data: {mean_squared_error(self.y_test, self.reg.predict(self.X_test))}')
            print(f'The Confusion Matrix: {confusion_matrix(self.y_test, self.reg.predict(self.X_test))}')
            print(f'The Classification report: {classification_report(self.y_test, self.reg.predict(self.X_test))}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')

    def best_k_value(self):
        try:
            self.train_accuracy = []
            self.k_values = np.arange(3, 50, 2)
            for i in self.k_values:
                self.reg = KNeighborsClassifier(n_neighbors=i)
                self.reg.fit(self.X_train, self.y_train)
                self.train_accuracy.append(self.reg.score(self.X_train, self.y_train))
                self.best_k_value = self.k_values[self.train_accuracy.index(max(self.train_accuracy))]
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')

    def training_with_best_k_value(self):
        try:
            self.reg_k = KNeighborsClassifier(n_neighbors=self.best_k_value)
            self.reg_k.fit(self.X_train, self.y_train)
            print(f'The Accuracy of Training data: {self.reg_k.score(self.X_train, self.y_train)}')
            print(f'The MSE of Training data: {mean_squared_error(self.y_train, self.reg_k.predict(self.X_train))}')
            print(f'The Confusion Matrix: {confusion_matrix(self.y_train, self.reg_k.predict(self.X_train))}')
            print(f'The Classification report: {classification_report(self.y_train, self.reg_k.predict(self.X_train))}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')

    def testing_with_best_k_value(self):
        try:
            print(f'The Accuracy of Testing data: {self.reg_k.score(self.X_test, self.y_test)}')
            print(f'The MSE of Testing data: {mean_squared_error(self.y_test, self.reg_k.predict(self.X_test))}')
            print(f'The Confusion Matrix: {confusion_matrix(self.y_test, self.reg_k.predict(self.X_test))}')
            print(f'The Classification report: {classification_report(self.y_test, self.reg_k.predict(self.X_test))}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'The Error Type: {er_type}')
            print(f'The Error Msg: {er_msg}')
            print(f'The Error Line no: {er_line.tb_lineno}')


if __name__ == '__main__':
    try:
        knn = KNN('breast_cancer.csv')
        knn.training_data()
        knn.testing_data()
        knn.best_k_value()
        knn.training_with_best_k_value()
        knn.testing_with_best_k_value()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'The Error Type: {er_type}')
        print(f'The Error Msg: {er_msg}')
        print(f'The Error Line no: {er_line.tb_lineno}')



