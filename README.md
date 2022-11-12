# Backorder Prediction

## Problem Statement
Backorders are unavoidable, but by anticipating which things will be backordered, planning can be streamlined at several levels, preventing unexpected strain on production, logistics, and transportation. ERP systems generate a lot of data (mainly structured) and also contain a lot of historical data; if this data can be properly utilized, a predictive model to forecast backorders and plan accordingly can be constructed. Based on past data from inventories, supply chain, and sales, classify the products as going into backorder (Yes or No).

## Project Objective
The objective of this project is to create a solution for above problem which can predict whether product will go on to be Backorder.

app link: https://backorder-prediction-ml.herokuapp.com/

## Project Demo Video
link: 

## Project Architecture
We have used layered architecture for carrying out below flow actions:
!(img)[https://github.com/shivanshka/Backorder-Prediction/blob/main/Architecture%20Design.png]

## Tools Used
- Jupyter Notebook
- VS Code
- Flask
- Machine Learning Algorithms: Balanced Random Forest Classifeir and Easy Ensemble Classifier
- MLOps
- HTML

## Dataset
We have taken data from Kaggle. It was a historical data with around 1.6 million datapoints in training dataset and 30,000 datapoints in Test dataset.

data link: https://github.com/rodrigosantis1/backorder_prediction/blob/master/dataset.rar

## Project Details
There are six packages in the pipeline: Config, Entity, Constant, Exception, Logger, Components and Pipeline

### Config
This package will create all folder structures and provide inputs to the each of the components.

### Entity
This package will defines named tuple for each of the components config and artifacts it generates.

### Constant
This package will contain all predefined constants which can be used accessed from anywhere

### Exception
This package contains the custom exception class for the Prediction Appliaction

### Logger
This package helps in logging all the activity

### Components
This package contains five modules:
1. Data Ingestion: This module downloads the data from the link, unzip it, then stores entire data into Db.
                   From DB it extracts all data into single csv file and split it into training and testing datasets.
2. Data Validation: This module validates whether data files passed are as per defined schema which was agreed upon
                    by client.
3. Data Transformation: This module applies all the Feature Engineering and preprocessing to the data we need to 
                        train our model and save  the pickle object for same.
4. Model Trainer: This module trains the model on transformed data, evalutes it based on R2 accuracy score and 
                  saves the best performing model object for prediction

### Pipeline
This package contains two modules:
1. Training Pipeline: This module will initiate the training pipeline where each of the above mentioned components  
                      will be called sequentially untill model is saved.
2. Prediction Pipeline: This module will help getting prediction from saved trained model.

## Contributors
Shivansh Kaushal