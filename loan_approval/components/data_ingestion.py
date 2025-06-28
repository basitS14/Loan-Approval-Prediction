from dataclasses import dataclass
from loan_approval.logger import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from loan_approval.exception import CustomException
from loan_approval.utils.main_utils import clean_data 

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts' , 'train.csv')
    test_data_path :str = os.path.join('artifacts' , 'test.csv')
    raw_data_path :str = os.path.join('artifacts' , 'data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            df = pd.read_csv('loan_approval/data/Copy of loan - loan.csv')
            print(df.head())
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)
            cleaned_df = clean_data(df)

            train_set , test_set = train_test_split(cleaned_df , train_size=0.8 , random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False , header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False , header=True)

            logging.info("Train-test split completed successfully")
            
            return (self.ingestion_config.train_data_path , self.ingestion_config.test_data_path)

        except Exception as e:
            print(CustomException(e , sys))


