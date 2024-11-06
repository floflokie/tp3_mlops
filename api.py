import flask
from load_model import load_model
import pandas as pd
app = flask.Flask(__name__)
model_name = "iris_model"
model_version = "latest"
model = load_model(model_name=model_name, model_version=model_version)

@app.route("/predict", methods=["POST"])
def predict():
    data = flask.request.json
    data = pd.DataFrame(data, index=[0])
    prediction = model.predict(data)
    return flask.jsonify({"prediction": prediction.tolist()[0]})

@app.route("/update-model", methods=["POST"])
def update_model():
    global model
    data = flask.request.json
    model_name = data["model_name"]
    model_version = data["model_version"]
    model_load = load_model(model_name=model_name, model_version=model_version)
    if model_load is None:
        return flask.jsonify({"message": "Model not found"})
    model = model_load
    return flask.jsonify({"message": f"Model updated to version {model}"})
