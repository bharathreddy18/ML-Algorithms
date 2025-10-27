import os
from abc import ABC, abstractmethod
import pandas as pd
import zipfile

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data froma given file."""
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content into pandas dataframe"""
        # Ensure the file is a .zip
        if not file_path.endswith('.zip'):
            raise ValueError('The provided file is not a .zip file')
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('extracted_data')

        # Find the extracted csv file
        extracted_files = os.listdir('extracted_data')
        csv_files = [i for i in extracted_files if i.endswith('.csv')]

        if len(csv_files) == 0:
            raise FileNotFoundError('No csv file in extracted data.')
        
        if len(csv_files) > 1:
            raise ValueError('Multiple csv files found. Please specify which one to use.')
        
        # Read the csv into Dataframe
        csv_file_path = os.path.join('extracted_data', csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df

# Implement a factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extention: str) -> DataIngestor:
        # returns appropriate data ingestor based on file extention.
        if file_extention == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f'No ingestor available for file extention: {file_extention}')
        
# Example usage:
if __name__ == ("__main__"):
    # Specify the file path
    file_path = 'C:\\Users\\Admin\\Desktop\\ML Projects\\House Price Prediction System\\archive.zip'

    # Determine the file extention
    file_extention = os.path.splitext(file_path)[1]

    # Get the appropriate Data Ingestor
    data_ingestor = DataIngestorFactory().get_data_ingestor(file_extention)

    # Ingest the data and get dataframe
    df = data_ingestor.ingest(file_path)

    # Now df will have csv file. lets check
    print(df.head())