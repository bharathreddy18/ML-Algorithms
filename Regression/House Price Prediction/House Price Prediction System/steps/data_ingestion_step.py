import pandas as pd
from src.data_ingestion import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    # Ingest data from a zip file using appropriate data ingestor method.
    # Determine file extention
    file_extention = '.zip'

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extention)
    df = data_ingestor.ingest(file_path)
    return df