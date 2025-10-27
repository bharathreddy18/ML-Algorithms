import pandas as pd
import numpy as np
import seaborn as sns
from abc import ABC, abstractmethod
import matplotlib.pylab as plt

# Abstract base class for univariate strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        '''
        Perform univariate analysis on a specific feature of the dataframe
        '''
        pass

# Concrete strategy for numerical features.
# This strategy analyzes numerical feature by plotting their distribution.

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=20)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

# Concrete strategy for categorical features
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue = feature, data=df, palette='muted')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

class UnivariateAnalyzer:
    def __init__(self, method):
        self.method = method

    def set_analyzer(self, method):
        self.method = method

    def execute_analyzer(self, df, feature):
        self.method.analyze(df, feature)

# if __name__ == "__main__":
#     univariate_analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
#     univariate_analyzer.execute_analyzer(df, feature)

