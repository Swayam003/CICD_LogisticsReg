from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import yaml
import joblib

params_path = "params.yaml"

app = Flask(__name__) # initializing a flask app

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    scalar_dir_path = config["webapp_scalar_dir"]
    scalr_model = joblib.load(scalar_dir_path)
    scalar_data = scalr_model.transform(data)
    model = joblib.load(model_dir_path)
    prediction = model.predict(scalar_data).tolist()[0]
    return  prediction

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            data = dict(request.form).values()
            data = [list(map(float, data))]
            prediction = predict(data)

            if prediction == 0:
                return render_template('results.html',predict = ("No, You dont have diabetes."))
            elif prediction == 1:
                return render_template('results.html',predict = ("Yes, You have diabetes."))
        except Exception as e:
            print('The Exception message is: ',e)
            error = {"error": "Something went Wrong!! Try again"}
            return render_template("404.html", error=error)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app
