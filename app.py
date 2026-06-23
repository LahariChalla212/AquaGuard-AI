from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    ph = float(request.form["ph"])
    hardness = float(request.form["hardness"])
    solids = float(request.form["solids"])
    chloramines = float(request.form["chloramines"])
    sulfate = float(request.form["sulfate"])
    conductivity = float(request.form["conductivity"])
    organic_carbon = float(request.form["organic_carbon"])
    trihalomethanes = float(request.form["trihalomethanes"])
    turbidity = float(request.form["turbidity"])

    features = [[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]]

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    # Confidence score
    confidence = max(model.predict_proba(features_scaled)[0]) * 100

    # Warning system
    warnings = []

    if ph < 6.5 or ph > 8.5:
        warnings.append("Abnormal pH Level")

    if turbidity > 5:
        warnings.append("High Turbidity")

    if sulfate > 400:
        warnings.append("High Sulfate Level")

    if chloramines > 4:
        warnings.append("High Chloramines")

    if conductivity > 500:
        warnings.append("High Conductivity")

    # Safe / Unsafe
    if prediction == 1:
        result = "✅ Safe to Drink"
        status = "safe"
    else:
        result = "❌ Not Safe to Drink"
        status = "unsafe"

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        status=status,
        warnings=warnings
    )


if __name__ == "__main__":
    app.run(debug=True)