import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

# Load dataset
df = pd.read_csv("btc.csv")

# Convert 'Date' to ordinal (numeric format)
df["Date"] = pd.to_datetime(df["Date"]).map(datetime.toordinal)

# Features and target variable
X = df[["Date"]]
y = df["Close"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    date_str = request.form["date"]
    date_num = datetime.strptime(date_str, "%Y-%m-%d").toordinal()
    prediction = model.predict(np.array([[date_num]]))[0]

    return jsonify({"date": date_str, "predicted_price": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
