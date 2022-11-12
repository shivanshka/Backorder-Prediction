
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

    """def missing_indicators(self,dff:pd.DataFrame)->list:
        missing_ind_num = []
        missing_ind_cat = []
        for feature, val in dff.isnull().sum().to_dict().items():
            if val != 0:
                if feature in self.dataset_schema[NUMERICAL_COLUMN_KEY]:
                    missing_ind_num.append(f"{feature}_missing_indicator")
                else:
                    missing_ind_cat.append(f"{feature}_missing_indicator")
        return (missing_ind_num,missing_ind_cat)"""

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
                
                logging.info("Feature Engineering applied !!!")
                featured_eng_data = self.fe_obj.transform(data_df)

                #missing_indicators = self.missing_indicators(featured_eng_data)
                
                numerical_columns = self.dataset_schema[NUMERICAL_COLUMN_KEY] 
                categorical_columns = self.dataset_schema[CATEGORICAL_COLUMN_KEY]

                
                # Applying preprocessing object on the data
                cols = numerical_columns+categorical_columns

                transformed_data= pd.DataFrame(self.preprocessing_obj.transform(featured_eng_data),columns=cols)
                logging.info("Data Preprocessing Done!!!")
                # Convertng datatype of feature accordingly
                transformed_data=transformed_data.infer_objects()

                # Predicting from the saved model object
                prediction_prob_arr = self.model_obj.predict_proba(transformed_data)[:,1]
                data_df["Will_go_on_Backorder"] = np.where(prediction_prob_arr>THRESHOLD,1,0)
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

            # Applying preprocessing object on the data
            preprocessed_df = pd.DataFrame(self.preprocessing_obj.transform(),columns=df.columns)
            
            # Changing datatype of features accordingly
            preprocessed_df = preprocessed_df.infer_objects()

            # Predicting from the saved model
            prediction_prob = self.model_obj.predict_proba(preprocessed_df)

            if prediction_prob >= THRESHOLD:
                prediction = "Product will go on BACKORDER!!!"
            else:
                prediction = "Product will NOT go on BACKORDER!!!"
            logging.info(f"{'*'*20} Single Prediction Complete {'*'*20}")
            return prediction
        except Exception as e:
            raise ApplicationException(e,sys) from e