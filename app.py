from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model          = pickle.load(open("model/best_model.pkl",      "rb"))
label_encoder  = pickle.load(open("model/label_encoder.pkl",   "rb"))
feature_columns= pickle.load(open("model/feature_columns.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html", symptoms=feature_columns)

@app.route("/predict", methods=["POST"])
def predict():
    selected = request.json.get("symptoms", [])
    vec = np.zeros(len(feature_columns))
    for s in selected:
        if s in feature_columns:
            vec[feature_columns.index(s)] = 1

    proba   = model.predict_proba([vec])[0]
    top3    = np.argsort(proba)[::-1][:3]
    results = [
        {"disease": label_encoder.inverse_transform([i])[0],
         "confidence": round(float(proba[i]) * 100, 2)}
        for i in top3
    ]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)