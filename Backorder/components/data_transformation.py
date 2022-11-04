
from Backorder.entity.config_entity import DataTransformationConfig
from Backorder.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from Backorder.contants import *
from Backorder.exception import ApplicationException
from Backorder.logger import logging
from Backorder.util.util import read_yaml_file, save_data, save_object
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os, sys

class Feature_Engineering(BaseEstimator,TransformerMixin):

    def __init__(self):
        try:
            logging.info(f"\n{'*'*20} Feature Engineering Started {'*'*20}\n\n")
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            
            deck_risk_freq_encoding = X["deck_risk"].value_counts(normalize=True).to_dict()
            X["deck_risk"] = X["deck_risk"].map(deck_risk_freq_encoding)

            ppap_risk_freq_encoding = X['ppap_risk'].value_counts(normalize=True).to_dict()
            X['ppap_risk'] = X['ppap_risk'].map(ppap_risk_freq_encoding)

            drop_columns=[]
            for feature in X.columns:
                if feature in ['sku','potential_issue','oe_constraint','stop_auto_buy','rev_stop']:
                    drop_columns.append(feature)
            
        
            X.drop(columns=drop_columns,axis=1,inplace = True)

            X['perf_6_month_avg'] = np.where(X['perf_6_month_avg'] == -99, np.nan, X['perf_6_month_avg'])
            X['perf_12_month_avg'] = np.where(X['perf_12_month_avg'] == -99, np.nan, X['perf_12_month_avg'])

            return X
        except Exception as e:
            raise ApplicationException(e,sys) from e

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"\n{'>>' * 30} Data Transformation log started {'<<' * 30}\n\n")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            data_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = data_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = data_schema[CATEGORICAL_COLUMN_KEY]

            num_pipeline = Pipeline(steps =[("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                            ("scaler",StandardScaler())])

            cat_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy="most_frequent",add_indicator=True)),
                                             ])

            preprocessing = ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
                                               ('cat_pipeline',cat_pipeline, categorical_columns)])
            return preprocessing
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def missing_indicators(self,dff)->list:
        missing_ind = []
        for feature, val in dff.isnull().sum().to_dict().items():
            if val != 0:
                missing_ind.append(f"{feature}_missing_indicator")

        return missing_ind

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path, low_memory=False)
            test_df = pd.read_csv(test_file_path, low_memory=False)

            # Reading schema file for columns details
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_file_path)

            # Extracting target column name
            target_column_name = schema[TARGET_COLUMN_KEY]

            train_df.dropna(subset=[target_column_name],inplace=True)
            test_df.dropna(subset=[target_column_name], inplace=True)

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns = target_column_name,axis = 1)
            target_feature_train_df = train_df[target_column_name]
            target_feature_train_df=target_feature_train_df.map({"Yes" : 1, "No" : 0})

            input_feature_test_df = test_df.drop(columns = target_column_name,axis = 1)
            target_feature_test_df = test_df[target_column_name]
            target_feature_test_df=target_feature_test_df.map({"Yes" : 1, "No" : 0})


            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            feature_eng_train_df = fe_obj.fit_transform(input_feature_train_df)
            feature_eng_test_df = fe_obj.transform(input_feature_test_df)

            train_missing_indicators = self.missing_indicators(feature_eng_train_df)
            test_missing_indicators = self.missing_indicators(feature_eng_test_df)

            numerical_columns = schema[NUMERICAL_COLUMN_KEY] 
            categorical_columns = schema[CATEGORICAL_COLUMN_KEY]

            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            train_arr = preprocessing_obj.fit_transform(feature_eng_train_df)
            test_arr = preprocessing_obj.transform(feature_eng_test_df)

            transformed_train_df = pd.DataFrame(np.c_[train_arr,target_feature_train_df],
                                            columns=numerical_columns+train_missing_indicators+categorical_columns+[target_column_name])
            transformed_test_df = pd.DataFrame(np.c_[test_arr,target_feature_test_df],
                                            columns=numerical_columns+test_missing_indicators+categorical_columns+[target_column_name])

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")
            transformed_test_file_path = os.path.join(transformed_test_dir,"transformed_test.csv")

            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            save_data(file_path = transformed_test_file_path, data = transformed_test_df)

            logging.info("Saving Feature Engineering Object")
            feature_eng_object_file_path = self.data_transformation_config.feature_eng_object_file_path
            save_object(file_path = feature_eng_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY, DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY,
                                os.path.basename(feature_eng_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY, DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                                            is_transformed=True,
                                            message="Data transformation successfull.",
                                            transformed_train_file_path = transformed_train_file_path,
                                            transformed_test_file_path = transformed_test_file_path,
                                            preprocessed_object_file_path = preprocessing_object_file_path,
                                            feature_eng_object_file_path = feature_eng_object_file_path)

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>>' * 30}Data Transformation log completed.{'<<' * 30}\n\n")