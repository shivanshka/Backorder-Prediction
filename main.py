
from flask import Flask, render_template,request, send_file, redirect,url_for,flash
from flask_cors import CORS, cross_origin
from Backorder.pipeline.prediction_pipeline import PredictionServices
from Backorder.pipeline.training_pipeline import Training_Pipeline
from Backorder.contants import *
from Backorder.util.util import read_yaml_file
from Backorder.logger import logging
import os
import sys
import shutil

app = Flask(__name__)
CORS(app)
app.secret_key = APP_SECRET_KEY

@app.route("/", methods =["GET"])
@cross_origin()
def home():
    return render_template("result.html")

@app.route("/bulk_predict", methods =["POST"])
@cross_origin()
def bulk_predict():
    try:
        file = request.files.get("files")
        folder = PREDICTION_DATA_SAVING_FOLDER_KEY

        flash("File uploaded!!","success")

        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

        file.save(os.path.join(folder,file.filename))

        pred = PredictionServices()
        output_file = pred.initiate_bulk_prediction()
        path = os.path.basename(output_file)

        flash("Prediction File generated!!","success")
        return send_file(output_file,as_attachment=True)

    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))

@app.route("/single_predict", methods =["POST"])
@cross_origin()
def single_predict():
    try:   
        data={}
        schema = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        for feature in schema[NUMERICAL_COLUMN_KEY]+schema[CATEGORICAL_COLUMN_KEY]:
            if feature.endswith("avg") or feature.endswith("risk"):
                data[feature] = float(request.form[feature])
            else:
                data[feature] = int(request.form[feature])

        pred = PredictionServices()
        output = pred.initiate_single_prediction(data)
        flash(output,"success")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))
    

@app.route("/start_train", methods=['GET', 'POST'])
@cross_origin()
def trainRouteClient():
    try:
        train_obj = Training_Pipeline()
        train_obj.run_training_pipeline() # training the model for the files in the table
    except Exception as e:
        flash(f'Something went wrong: {e}', 'danger')
        logging.error(e)
        return redirect(url_for('home'))


if __name__=="__main__":
    
    port = int(os.getenv("PORT",5003))
    host = '0.0.0.0'
    app.run(host=host,port=port,debug=True)
    
    