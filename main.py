from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return "Application is under construction"



if __name__=="__main__":
    app.run()