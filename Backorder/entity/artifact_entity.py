from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",["train_file_path",
                                                            "test_file_path",
                                                            "is_ingested",
                                                            "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",["schema_file_path",
                                                              "report_file_path",
                                                              "report_page_file_path",
                                                              "is_validated","message"]) 

DataTransformationArtifact = namedtuple("DataTransformationArtifact",["is_transformed", 
                                                                      "message", 
                                                                      "transformed_train_file_path",
                                                                      "transformed_test_file_path",
                                                                      "preprocessed_object_file_path",
                                                                      "feature_eng_object_file_path"])      

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", 
                                                            "message", 
                                                            "trained_model_file_path",
                                                            "roc_auc_score",
                                                            "threshold"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pushed", "export_model_file_path"])