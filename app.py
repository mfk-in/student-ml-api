from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    study = data["study_hours"]
    sleep = data["sleep_hours"]
    attendance = data["attendance"]

    prediction = model.predict(
        np.array([[study, sleep, attendance]])
    )

    return jsonify({
        "predicted_score" : round(prediction[0], 2)
    })
if __name__ == "__main__":
    app.run(debug=True)