from time import time
from Backorder.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, \
    ModelTrainerConfig,ModelEvaluationConfig, ModelPusherConfig, TrainingPipelineConfig
from Backorder.contants import *
from Backorder.util.util import read_yaml_file
from Backorder.exception import ApplicationException
from Backorder.logger import logging
import os, sys

class Configuration:
    def __init__(self,config_file_path:str = CONFIG_FILE_PATH, time_stamp:str = CURRENT_TIME_STAMP)-> None:
        try:
            self.config_info = read_yaml_file(file_path = config_file_path)
            self.time_stamp = time_stamp
            self.training_pipeline_config = self.get_training_pipeline_config()
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir=os.path.join(artifact_dir,
                                                     DATA_INGESTION_ARTIFACT_DIR_KEY,
                                                     self.time_stamp)
            
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            zipped_download_dir = os.path.join(data_ingestion_artifact_dir,
                                            data_ingestion_info[DATA_INGESTION_ZIPPED_DOWNLOAD_DIR_KEY])
            
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])

            ingested_data_dir = os.path.join(data_ingestion_artifact_dir,
                                             data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            
            ingested_train_dir = os.path.join(ingested_data_dir,
                                              data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])
            
            ingested_test_dir =os.path.join(ingested_data_dir,
                                            data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])


            data_ingestion_config=DataIngestionConfig(dataset_download_url=dataset_download_url, 
                                                      zipped_download_dir=zipped_download_dir, 
                                                      raw_data_dir=raw_data_dir, 
                                                      ingested_train_dir=ingested_train_dir, 
                                                      ingested_test_dir=ingested_test_dir)

            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_data_validation_config(self)-> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_validation_artifact_dir = os.path.join(artifact_dir,
                                                        DATA_VALIDATION_ARTIFACT_DIR_KEY,
                                                        self.time_stamp)
            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            schema_file_path = os.path.join(ROOT_DIR,
                                            data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                            data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])

            report_file_path = os.path.join(data_validation_artifact_dir,
                                            data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY])

            report_page_file_path = os.path.join(data_validation_artifact_dir,
                                                 data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])

            data_validation_config = DataValidationConfig(schema_file_path=schema_file_path,
                                                         report_file_path = report_file_path,
                                                         report_page_file_path=report_page_file_path)
            logging.info(f"Data Validation config: {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_data_transformation_config(self)-> DataTransformationConfig:
        try:
            pass
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])

            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            logging.info(f"Training Pipeline Config : {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise ApplicationException(e,sys) from e