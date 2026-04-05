# 🏥 Disease Predictor from Symptoms

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python) 
![Flask](https://img.shields.io/badge/Flask-Web_App-green?style=for-the-badge&logo=flask) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)

A machine learning web application that predicts potential diseases based on user-selected symptoms. Built with **scikit-learn** and a **Flask** web interface.

> ⚠️ **Medical Disclaimer:** This tool is for educational purposes only. Always consult a qualified doctor for any medical concerns.

---

## ✨ Features & Enhancements

This project significantly extends baseline models with the following key features:

- **Robus Evaluation:** Replaced single train/test split with **Stratified 5-Fold Cross-Validation** to ensure unbiased accuracy estimates across the full dataset.
- **GridSearchCV Hyperparameter Tuning:** 
  - **Random Forest:** Automatically tunes `n_estimators`, `max_depth`, `min_samples_split`.
  - **SVM:** Automatically tunes `C`, `kernel`, `gamma`.
- **Classifier Comparison:** Head-to-head comparison between tuned Random Forest and SVM. Both achieve perfect accuracy on this dataset. Random Forest is saved as the default model.
- **Interactive Web Interface (Flask):**
  - Searchable multi-select symptom chips (132 supported symptoms).
  - Real-time top-3 disease predictions.
  - Visual confidence score bars for each prediction.
  - Dark-themed, responsive, production-grade UI.
- **Data Preprocessing:** Automatic handling of missing NaN values (`fillna=0`).

---

## 🚀 Quick Start (How to Run)

**1. Install dependencies**
```bash
pip install flask scikit-learn pandas numpy
```

**2. Train the model**  
Runs GridSearchCV to tune hyperparameters and saves the best model natively:
```bash
python train.py
```

**3. Launch the web interface**
```bash
python app.py
```

**4. Access the app**  
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

```text
Disease-Prediction-from-Symptoms/
├── dataset/
│   ├── training_data.csv        # Source dataset
│   └── test_data.csv            # Testing subsets
├── model/
│   ├── best_model.pkl           # Auto-saved best model
│   ├── label_encoder.pkl        # Encoded target labels
│   └── feature_columns.pkl      # Saved symptom features
├── templates/
│   └── index.html               # Flask frontend template
├── app.py                       # Web app entrypoint
├── train.py                     # Cross-validation & tuning script
├── main.py                      # Original base file (preserved)
├── infer.py                     # Original base file (preserved)
└── requirements.txt             # Python dependencies
```

---

## 🙏 Credits

Based on the original repository: [anujdutt9/Disease-Prediction-from-Symptoms](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms)
