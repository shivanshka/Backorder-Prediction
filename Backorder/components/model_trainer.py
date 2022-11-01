import os,sys
import pandas as pd
import numpy as np
from Backorder.contants import *
from Backorder.util.util import save_object
from Backorder.entity.config_entity import ModelTrainerConfig
from Backorder.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from Backorder.logger import logging
from Backorder.exception import ApplicationException
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.model_selection import GridSearchCV


class Model_Trainer:
    def __init__(self, model_trainer_config : ModelTrainerConfig, 
                       data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'>>>'*20} Model Training Started {'<<<'*20}\n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_best_params_EasyEnsemble(self,X_train, y_train):
        try:
            logging.info("Running Hyperparameter Tuning for Easy Ensemble Classifier")
            params = {"n_estimators" : [10, 20, 30, 50, 100],"sampling_strategy":["auto"]}
            easy_gs = GridSearchCV(EasyEnsembleClassifier(random_state=2021), 
                 param_grid=params, scoring="roc_auc", cv=5, refit=True, verbose=False)
            easy_gs.fit(X_train, y_train)
            best_params = easy_gs.best_params_
            return best_params
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_best_params_BalancedRF(self,X_train, y_train):
        try:
            logging.info("Running Hyperparameter Tuning for Balanced Random Forest Classifier")
            params = {"n_estimators" : [10, 20, 30, 50, 100],
                      "max_depth" : [3,5,7,9,11,13,15],
                      "criterion": ["gini"],
                      "sampling_strategy": ["auto"]}
            brf_grid = GridSearchCV(BalancedRandomForestClassifier(random_state=2021), 
                 param_grid=params, scoring="roc_auc", cv=5, refit=True, verbose=False)
            brf_grid.fit(X_train, y_train)

            best_params = brf_grid.best_params_
            return best_params
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def EasyEnsemble_Classifier(self,X_train, y_train):
        try:
            logging.info("Getting best parameters for EasyEnsemble Classifier")
            params = self.get_best_params_EasyEnsemble(X_train, y_train)

            logging.info(f"Grid search completed. Best params : {params}")

            logging.info("Fitting EasyEnsemble Classifier model.......")
            skf = StratifiedKFold(n_splits = 5, random_state = 2021, shuffle = True)
            train_auc_scores = []
            test_auc_scores = []
            
            for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                x_train, Y_train  = X_train.iloc[train_index], y_train[train_index]
                x_test, Y_test = X_train.iloc[test_index], y_train[test_index]
                
                model = EasyEnsembleClassifier(random_state=2021, **params)
                model.fit(x_train, Y_train)
                
                train_preds = model.predict_proba(x_train)[:,1]
                test_preds = model.predict_proba(x_test)[:,1]
                train_auc = roc_auc_score(Y_train, train_preds)
                test_auc = roc_auc_score(Y_test, test_preds)
                train_auc_scores.append(train_auc)
                test_auc_scores.append(test_auc) 
            
            logging.info(f"Train Mean ROC AUC Score for 5 fold : {np.mean(train_auc_scores)}")
            logging.info(f"Validation Mean ROC AUC Score for 5 fold : {np.mean(test_auc_scores)}")
            return model
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def BalancedRF_Classifier(self,X_train, y_train):
        try:
            logging.info("Getting best parameters for Balanced RandomForest Classifier")
            params = self.get_best_params_BalancedRF(X_train, y_train)

            logging.info(f"Grid search completed. Best params : {params}")

            logging.info("Fitting Balanced RandomForest Classifier model.......")
            skf = StratifiedKFold(n_splits = 5, random_state = 2021, shuffle = True)
            train_auc_scores = []
            test_auc_scores = []
            
            for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                x_train, Y_train  = X_train.iloc[train_index], y_train[train_index]
                x_test, Y_test = X_train.iloc[test_index], y_train[test_index]
                
                model = BalancedRandomForestClassifier(random_state=2021, **params)
                model.fit(x_train, Y_train)
                
                train_preds = model.predict_proba(x_train)[:,1]
                test_preds = model.predict_proba(x_test)[:,1]
                train_auc = roc_auc_score(Y_train, train_preds)
                test_auc = roc_auc_score(Y_test, test_preds)
                train_auc_scores.append(train_auc)
                test_auc_scores.append(test_auc)
            
            logging.info(f"Train Mean ROC AUC Score for 5 fold : {np.mean(train_auc_scores)}")
            logging.info(f"Validation Mean ROC AUC Score for 5 fold : {np.mean(test_auc_scores)}")
            return model
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def best_model_selector(self, X_train, y_train, X_test, y_test):
        try:
            logging.info(f'{"*"*20} Training Easy Ensemble Classifier {"*"*20}')
            easy_ensemble = self.EasyEnsemble_Classifier(X_train, y_train)
            
            logging.info(f"Calculating Test ROC AUC Score..........")
            easy_test_preds = easy_ensemble.predict_proba(X_test)[:,1]
            easy_ensemble_test_auc_score = roc_auc_score(y_test, easy_test_preds)
            logging.info(f"Test ROC AUC Score: {easy_ensemble_test_auc_score}")

            logging.info(f'{"*"*20} Training Easy Ensemble Classifier Completed Successfully!! {"*"*20}')

            logging.info(f'{"*"*20} Training Balanced RandomForest Classifier {"*"*20}')
            balanced_rf = self.BalancedRF_Classifier(X_train, y_train)

            logging.info(f"Calculating Test ROC AUC Score..........")
            brf_test_preds = balanced_rf.predict_proba(X_test)[:,1]
            balanced_rf_test_auc_score = roc_auc_score(y_test, brf_test_preds)
            logging.info(f"Test ROC AUC Score: {balanced_rf_test_auc_score}")

            logging.info(f'{"*"*20} Training Balanced RandomForest Classifier Completed Successfully!! {"*"*20}')

            if easy_ensemble_test_auc_score > balanced_rf_test_auc_score:
                best_model = easy_ensemble
            else:
                best_model = balanced_rf

            return best_model
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def initiate_model_training(self):
        try:
            logging.info("Finding transformed Training and Test")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Transformed Data found!!! Now, converting it into dataframe")
            train_df = pd.read_csv(transformed_train_file_path)
            test_df = pd.read_csv(transformed_test_file_path)

            train_df = train_df.infer_objects()
            test_df = test_df.infer_objects()

            logging.info("Splitting Input features and Target Feature for train and test data")
            train_input_feature = train_df.iloc[:,:-1]
            train_target_feature = train_df.iloc[:,-1]

            test_input_feature = test_df.iloc[:,:-1]
            test_target_feature = test_df.iloc[:,-1]

            trained_model = self.best_model_selector(train_input_feature, train_target_feature, test_input_feature, test_target_feature)
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path

            save_object(trained_model_file_path, trained_model)
            #save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
            #                     os.path.basename(trained_model_object_file_path)),obj=model_obj)

            model_training_artifact = ModelTrainerArtifact(is_trained= "Yes", 
                                                           message= "Model Training Done!", 
                                                           trained_model_file_path=trained_model_file_path,
                                                           )
            logging.info(f"Model Training Artifact : [{model_training_artifact}]")
            return model_training_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>>>'*20} Model Training Completed {'<<<'*20}\n\n")

