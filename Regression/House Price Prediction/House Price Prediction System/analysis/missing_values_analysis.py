import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Abstract Base class for Missing values Analysis.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass

# Concrete class for missing values identification.
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print('\nMissing values count by column:')
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])

    def visualize_missing_values(self, df: pd.DataFrame):
        print('\nVisualizing Missing Values...')
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
        plt.title('Missing values Heatmap')
        plt.show()

if __name__ == "__main__":
    data_path = 'C:\\Users\\Admin\\Desktop\\ML Projects\\House Price Prediction System\\extracted_data\\AmesHousing.csv'
    df = pd.read_csv(data_path)
    missing = SimpleMissingValuesAnalysis()
    missing.analyze(df)