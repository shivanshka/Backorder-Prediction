
from flask import Flask, render_template,request, send_file, redirect,url_for,flash
from flask_cors import CORS, cross_origin
#from Prediction_Application.pipeline.prediction_pipeline import Prediction
from Backorder.pipeline.training_pipeline import Training_Pipeline
from Backorder.constant import *
from Backorder.logger import logging
import os
import sys
import shutil

app = Flask(__name__)
CORS(app)
#app.secret_key = APP_SECRET_KEY

@app.route("/", methods =["GET"])
@cross_origin()
def home():
    return render_template("result.html")

@app.route("/bulk_predict", methods =["POST"])
@cross_origin()
def bulk_predict():
    try:
        file = request.files.get("files")
        #folder = PREDICTION_DATA_SAVING_FOLDER_KEY

        flash("File uploaded!!","success")

        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

        file.save(os.path.join(folder,file.filename))

        pred = Prediction()
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
        data = {'date': request.form['date'],
                'month': int(request.form['month']),
                'hour': int(request.form['hour']),
                'season': int(request.form['season']),
                'weekday': int(request.form['weekday']),
                'is_holiday': int(request.form['is_holiday']),
                'working_day': int(request.form['working']),
                'weather_sit': int(request.form['weather_sit']),
                'is_covid': int(request.form['is_covid']),
                'temp': float(request.form['temp']),
                'wind': float(request.form['wind']),
                'humidity': float(request.form['humidity'])}

        pred = Prediction()
        output = pred.initiate_single_prediction(data)
        flash(f"Predicted Demand for Bike for given conditions: {output}","success")
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
    
    