
from Backorder.components.data_validation import Prediction_Validation
from Backorder.config.configuration import Configuration
from Backorder.logger import logging
from Backorder.exception import ApplicationException
from Backorder.logger import logging
from Backorder.contants import *
from Backorder.util.util import load_object, read_yaml_file, save_data
import os,sys, shutil
import pandas as pd
import numpy as np

class PredictionServices:

    def __init__(self,config:Configuration = Configuration()):
        """
        Prediction Class : It helps in predicting from saved trained model.
                           It has two modes Bulk Prediction and Single Prediction

            created by:
                    Shivansh Kaushal and Chirag Sharma
        """


        logging.info(f"\n{'*'*20} Prediction Pipeline Initiated {'*'*20}\n")

        self.config_info = read_yaml_file(CONFIG_FILE_PATH)

        # Getting data validation config info
        self.data_validation_config = config.get_data_validation_config()
        self.data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

        # Loading Feature Engineering, Preprocessing and Model pickle objects for prediction
        self.fe_obj = load_object(file_path=os.path.join(ROOT_DIR,
                                            PIKLE_FOLDER_NAME_KEY,
                                            self.data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                            self.data_transformation_config_info[DATA_TRANSFORMATION_FEAT_ENG_FILE_NAME_KEY]))

        self.preprocessing_obj = load_object(file_path=os.path.join(ROOT_DIR,
                                        PIKLE_FOLDER_NAME_KEY,
                                        self.data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                        self.data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]))

        self.model_obj = load_object(file_path=os.path.join(ROOT_DIR,
                                                PIKLE_FOLDER_NAME_KEY,
                                                SERVING_MODEL_NAME_KEY,
                                                self.config_info[MODEL_TRAINER_CONFIG_KEY][MODEL_TRAINER_FILE_NAME_KEY]))

        # Reading schema.yaml file to validate prediction data
        self.schema_file_path = self.data_validation_config.schema_file_path
        self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)

    def initiate_bulk_prediction(self):
        """
        Function to predict from saved trained model for entire dataset. It returns the original dataset \n
        with prediction column
        """
        try:
            logging.info(f"{'*'*20}Bulk Prediction Mode Selected {'*'*20}")
            # Getting location of uploaded dataset
            self.folder = PREDICTION_DATA_SAVING_FOLDER_KEY
            self.path = os.path.join(self.folder,os.listdir(self.folder)[0])
            
            # Validating uploaded dataset
            logging.info(f"Validatiog Passed Dataset : [{self.path}]")
            pred_val = Prediction_Validation(self.path,self.data_validation_config)
            data_validation_status = pred_val.validate_dataset_schema()

            logging.info(f"Prediction for dataset: [{self.path}]")

            if data_validation_status:
                # Reading uploaded .CSV file in pandas
                data_df = pd.read_csv(self.path)
                data_df['total_count'] = 0
                col = ['date','year','month','hour','season','weekday','is_holiday','working_day','total_count',
                    'temp','wind','humidity','weather_sit','is_covid']
                
                logging.info("Feature Engineering applied !!!")
                featured_eng_data = pd.DataFrame(self.fe_obj.transform(data_df),columns=col)
                featured_eng_data.drop(columns="total_count", inplace=True)
                
                date_cols = featured_eng_data.loc[:,['date','year','month','hour']]

                cols = ['date','year','month','hour','season','weekday','is_holiday','working_day','weather_sit',
                'is_covid','temp','wind','humidity']
                data_df = data_df[~data_df.duplicated(subset=["date","month","hour"],keep='last')]
                data_df.drop(columns="total_count",inplace=True)

                logging.info("Data Preprocessing Done!!!")
                # Applying preprocessing object on the data
                transformed_data = pd.DataFrame(np.c_[date_cols,self.preprocessing_obj.transform(featured_eng_data)],columns=cols)
                
                transformed_data.drop(columns=["year"], inplace=True)
                transformed_data.set_index("date",inplace=True)
                
                # Convertng datatype of feature accordingly
                transformed_data=transformed_data.infer_objects()

                # Predicting from the saved model object
                prediction = self.model_obj.predict(transformed_data)
                data_df["predicted_demand"] = prediction
                logging.info("Prediction from model done")

                logging.info("Saving prediction file for sending it to the user")

                output_folder_file_path = os.path.join(ROOT_DIR,"Output Folder",CURRENT_TIME_STAMP,"Predicted.csv")
                if os.path.exists(os.path.join(ROOT_DIR,"Output Folder")):
                    shutil.rmtree(os.path.join(ROOT_DIR,"Output Folder"))

                save_data(file_path=output_folder_file_path,data = data_df)
                zipped_file = os.path.dirname(output_folder_file_path)
                
                shutil.make_archive(zipped_file,"zip",zipped_file)
                shutil.rmtree(zipped_file)
                shutil.rmtree(self.folder)
                
                logging.info(f"{'*'*20} Bulk Prediction Coomplete {'*'*20}")
                return zipped_file+".zip"

        except Exception as e:
            raise ApplicationException(e,sys) from e 

    def initiate_single_prediction(self,data:dict)->int:
        """
        Function to predict from the saved train model. It predicts from single value of each feature.
        """
        try:
            logging.info(f"{'*'*20} Single Prediction Mode Selected {'*'*20}")
            logging.info(f"Passed Info: [{data}]")
            
            # Converting passed data into DataFrame
            df = pd.DataFrame([data])
            date_cols = df.loc[:,["date","month","hour"]]

            # Applying preprocessing object on the data
            preprocessed_df = pd.DataFrame(np.c_[date_cols,self.preprocessing_obj.transform(df.drop(columns=["date","month","hour"]))],
            columns=df.columns)
            preprocessed_df.set_index("date",inplace=True)

            # Changing datatype of features accordingly
            preprocessed_df = preprocessed_df.infer_objects()

            # Predicting from the saved model
            prediction = self.model_obj.predict(preprocessed_df)
            logging.info(f"{'*'*20} Single Prediction Complete {'*'*20}")
            return round(prediction[0])
        except Exception as e:
            raise ApplicationException(e,sys) from e