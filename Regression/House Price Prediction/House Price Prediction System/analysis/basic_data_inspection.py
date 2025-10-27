import pandas as pd
from abc import ABC, abstractmethod

# Abstract base class for Data Inspection Strategies
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass

# Concrete strategy for Data Types Inspection
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print('\nData Types and Non-Null Counts.')
        print(df.info())

# Concrete strategy for summary statistics inspection.
class SummaryStatisticsDataInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print('\nSummary Statistics(Numerical Features):')
        print(df.describe())

        print('\nSummary Statistics(Categorical Features):')
        print(df.describe(include=['O']))

# Context class that uses data inspection strategy.
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self.strategy.inspect(df)

if __name__ == "__main__":
    df = pd.read_csv('C:\\Users\\Admin\\Desktop\\ML Projects\\House Price Prediction System\\extracted_data\\AmesHousing.csv')
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)

    inspector.set_strategy(SummaryStatisticsDataInspectionStrategy())
    inspector.execute_inspection(df)