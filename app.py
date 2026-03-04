from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)