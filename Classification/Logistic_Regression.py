import pandas as pd
import numpy as np
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix

class LR:
    def __init__(self):
        try:
            self.p = datasets.load_breast_cancer()
            self.df = pd.DataFrame(data=self.p['data'], columns=self.p['feature_names'])
            self.df['Cancer'] = self.p['target']  # 1 - Benign, 0 - Malignant
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}\n Error Value: {er_val}\n Er TB: {er_tb.tb_lineno}')

    def training(self):
        try:
            self.X_train,self.X_test, self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            self.lr = LogisticRegression()
            self.lr.fit(self.X_train, self.y_train)
            self.y_train_pred = self.lr.predict(self.X_train)
            print('----------------------TRAINING PERFORMANCE---------------------------')
            print(f'Accuracy Score: {accuracy_score(self.y_train, self.y_train_pred)}\n')
            print(f'MSE: {mean_squared_error(self.y_train, self.y_train_pred)}\n')
            print(f'Confusion Matrix: {confusion_matrix(self.y_train, self.y_train_pred)}\n')
            print(f'Classification Report: {classification_report(self.y_train, self.y_train_pred)}\n')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}\n Error Value: {er_val}\n Er TB: {er_tb.tb_lineno}')

    def testing(self):
        try:
            self.y_test_pred = self.lr.predict(self.X_test)
            print('-----------------------TESTING PERFORMANCE---------------------------')
            print(f'Accuracy Score: {accuracy_score(self.y_test, self.y_test_pred)}\n')
            print(f'MSE: {mean_squared_error(self.y_test, self.y_test_pred)}\n')
            print(f'Confusion Matrix: {confusion_matrix(self.y_test, self.y_test_pred)}\n')
            print(f'Classification Report: {classification_report(self.y_test, self.y_test_pred)}\n')
        except Exception as e:
            er_type, er_val, er_tb = sys.exc_info()
            print(f'Error type: {er_type}\n Error Value: {er_val}\n Er TB: {er_tb.tb_lineno}')


if __name__ == "__main__":
    try:
        lr = LR()
        lr.training()
        lr.testing()
    except Exception as e:
        er_type, er_val, er_tb = sys.exc_info()
        print(f'Error type: {er_type}\n Error Value: {er_val}\n Er TB: {er_tb.tb_lineno}')
