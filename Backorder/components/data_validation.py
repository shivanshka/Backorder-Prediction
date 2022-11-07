from Backorder.logger import logging
from Backorder.exception import ApplicationException
from Backorder.entity.config_entity import DataValidationConfig, DataIngestionConfig
from Backorder.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from Backorder.util.util import read_yaml_file
import os,sys
import pandas as pd
import re
import json
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

class Prediction_Validation:
    def __init__(self,path, data_validation_config : DataValidationConfig):
        try:
            self.path=path
            self.data_validation_config = data_validation_config
            self.schema_file_path = self.data_validation_config.schema_file_path
            self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def column_check(self,file):   
        try:
            data = pd.read_csv(file)
            # Finding no of columns in the dataset
            no_of_columns = data.shape[1]
            # Checking if the no of columns in dataset is as per defined schema
            if no_of_columns != self.dataset_schema["NumberOfColumns"]:
                raise Exception(f"No of columns is not correct in file: [{file}]!!!")

            columns = list(data.columns)

            # Checking for columns name , whether they are as per the defined schema
            for column in columns:
                if column not in self.dataset_schema["Columns"].keys():
                    raise Exception(f"Column :[{column}] in file: [{file}] not available in the Schema!!!")

            # Checking whether any column have entire rows as missing value
            count = 0
            col = []
            for column in columns:            
                if (len(data[column]) - data[column].count()) == len(data[column]):
                    count+=1
                    col.append(column)
            if count > 0:
                raise Exception(f"Columns: [{col}] have entire row as missing value") 

            return True

        except Exception as e:
            raise ApplicationException(e,sys) from e 
    
    def validate_dataset_schema(self):
        try:
            logging.info("Validating the schema of the dataset")
            validation_status = False
            
            if self.column_check(os.path.join(self.path)):
                validation_status = True
                
            logging.info("Schema Validation Completed")
            logging.info(f"Is dataset schema as per the defined schema? -> {validation_status}")
            return validation_status
            
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info("Prediction Dataset Validation log complete")

class DataValidation:
    def __init__(self, data_validation_config : DataValidationConfig,
                    data_ingestion_config : DataIngestionConfig, 
                    data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"\n{'>'*20} Data Validation log started {'<'*20}\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_config = data_ingestion_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_file_path = self.data_validation_config.schema_file_path
            self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_train_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path, low_memory=False)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path, low_memory=False)
            train_df = train_df.iloc[:-1,:]
            return train_df,test_df
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def is_train_test_file_exists(self)->bool:
        try: 
            logging.info("Checking if the training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file} is not present"
                raise Exception(message)

            return is_available
        
        except Exception as e:
            raise ApplicationException(e,sys) from e

        
    def column_check(self,file):   
        try:
            data = pd.read_csv(file)
            # Finding no of columns in the dataset
            no_of_columns = data.shape[1]
            # Checking if the no of columns in dataset is as per defined schema
            if no_of_columns != self.dataset_schema["NumberOfColumns"]:
                raise Exception(f"No of columns is not correct in file: [{file}]!!!")

            columns = list(data.columns)
            columns.remove(self.dataset_schema["target_column"])

            # Checking for columns name , whether they are as per the defined schema
            for column in columns:
                if column not in self.dataset_schema["Columns"].keys():
                    raise Exception(f"Column :[{column}] in file: [{file}] not available in the Schema!!!")

            # Checking whether any column have entire rows as missing value
            count = 0
            col = []
            for column in columns:            
                if (len(data[column]) - data[column].count()) == len(data[column]):
                    count+=1
                    col.append(column)
            if count > 0:
                raise Exception(f"Columns: [{col}] have entire row as missing value") 

            return True

        except Exception as e:
            raise ApplicationException(e,sys) from e 
    
    def validate_dataset_schema(self):
        try:
            logging.info("Validating the schema of the dataset")
            validation_status = False
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            file_name = os.listdir(raw_data_dir)[0]
            data_file_path = os.path.join(raw_data_dir,file_name)
            
            for file in os.listdir(data_file_path):
                if self.column_check(os.path.join(data_file_path,file)):
                    validation_status = True
                
            logging.info("Schema Validation Completed")
            logging.info(f"Is dataset schema as per the defined schema? -> {validation_status}")
            return validation_status
            
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_and_save_data_drift_report(self):
        try:
            logging.info("Generating data drift report.json file")
            profile = Profile(sections = [DataDriftProfileSection()])
            train_df, test_df = self.get_train_test_df()
            profile.calculate(train_df, test_df)
            
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent = 6)
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise ApplicationException(e,sys) from e   

    def save_data_drift_report_page(self):
        try:
            logging.info("Generating data drift report.html page")
            dashboard = Dashboard(tabs = [DataDriftTab()])
            train_df, test_df = self.get_train_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
            logging.info("Report.html page generation successful!!")
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            logging.info("Checking for Data Drift")
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.validate_dataset_schema()
            self.is_train_test_file_exists()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successfully.")

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>'*20} Data Validation log completed {'<'*20}\n")