from Backorder.logger import logging
from Backorder.exception import ApplicationException
from Backorder.entity.config_entity import DataIngestionConfig
from Backorder.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import numpy as np
import os,sys
from pyunpack import Archive
from six.moves import urllib



class DataIngestion:
    def __init__(self,data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"\n{'>'*20} Data Ingestion log started {'<'*20}\n")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ApplicationException(e,sys) from e

    def download_data(self):
        """
        Downloads the zipped dataset from the given url and save it to the specified path.
        """
        try:
            # Extracting remote url to download dataset files
            download_url = self.data_ingestion_config.dataset_download_url

            # folder location to download zipped file
            zipped_download_dir = self.data_ingestion_config.zipped_download_dir

            if os.path.exists(zipped_download_dir):
                os.remove(zipped_download_dir)
            os.makedirs(zipped_download_dir,exist_ok=True)

            #file_name = os.path.basename(download_url)
            file_name = "backorder.rar"
            zipped_file_path = os.path.join(zipped_download_dir,file_name)

            logging.info(f"Downloading file from: [{download_url}] into : [{zipped_file_path}]")
            urllib.request.urlretrieve(download_url,zipped_file_path)
            logging.info(f"File: [{zipped_file_path}] has been downloaded successfully")

            return zipped_file_path

        except Exception as e:
            raise ApplicationException(e,sys) from e

    def extract_zipped_file(self,zipped_file_path:str):
        try:
            # Folder location to extract the downloaded zipped dataset files
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir,exist_ok=True)
            
            logging.info(f"Extracting zipped file : [{zipped_file_path}] into dir: [{raw_data_dir}]")
            # Extarcting the files from zipped file
            Archive(zipped_file_path).extractall(raw_data_dir)
            logging.info("Extraction completed successfully")

        except Exception as e:
            raise ApplicationException(e,sys) from e

    def data_merge_and_split(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir  # Location for extracted data files
            
            file_name = os.listdir(raw_data_dir)[0]
            data_file_path = os.path.join(raw_data_dir,file_name)
            
            # Splitting the dataset into train and test data based on date indexing
            logging.info("Splitting Dataset into train and test")
            for file in os.listdir(data_file_path):
                if file.split(".")[0].endswith("train"):
                    train_set = pd.read_csv(os.path.join(data_file_path,file),low_memory=False)
                    train_set = train_set.iloc[:-1,:]
                else:
                    test_set = pd.read_csv(os.path.join(data_file_path,file),low_memory=False)
                    test_set = test_set.iloc[:-1,:]

            # Setting paths for train and test data
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,"train.csv")
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,"test.csv")

            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path,index=False)

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path,index=False)


            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message="Data ingestion completed successfully")
            logging.info(f"Data Ingestion Artifact: [{data_ingestion_artifact}]")
            return data_ingestion_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            zipped_file_path = self.download_data()
            self.extract_zipped_file(zipped_file_path=zipped_file_path)
            return self.data_merge_and_split()
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    def __del__(self):
        logging.info(f"\n{'>'*20} Data Ingestion log completed {'<'*20}\n")

