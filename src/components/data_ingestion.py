import os # for file/directory management
import sys # give access to system specific functions like custom exceptions
from src.exception import CustomException
from src.logger import logging
import pandas as pd # for data manipulation
from sklearn.model_selection import train_test_split # for train test split
from dataclasses import dataclass # simplifies class definitions used to store config

@dataclass
# This class simply holds file paths for where the ingested data will be stored
# All files go to artifacts folder
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

# This ensures that the rest of the class has access to all file paths easily
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()  

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") # Logs that ingestion has started
        try:
            data_path = os.path.join(os.getcwd(), "Notebooks", "data", "stud.csv")
            df = pd.read_csv(data_path) # reading the CSV file in DataFrame format
            logging.info('Read the dataset as dataframe') # Logging the event

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # create artifact folder if it doesn't exist

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # saves raw data inside data folder

            logging.info("Train test split initiated") # logging the event
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) # training the data

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # saving into specific file

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # saving into specific file

            logging.info("Ingestion of the data is completed") 

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path   # Returns paths to the train and test files so the next pipeline step can use them

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()