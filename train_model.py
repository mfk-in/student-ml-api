import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("student_data.csv")

X = data[["study_hours", "sleep_hours", "attendance"]]
Y = data["exam_score"]

model = LinearRegression()

model.fit(X,Y)

with open("model.pkl", "wb") as file:
    pickle.dump(model,file)

print("Model trained and saved successfully!")
