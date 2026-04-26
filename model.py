import pandas as pd
import numpy as np

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# 1. LOAD DATASET
# =====================================================--
df = pd.read_csv("hospital_readmissions_30k.csv")

# =====================================================
# 2. DATA CLEANING
# =====================================================
# Encode target
df['readmitted_30_days'] = df['readmitted_30_days'].map({
    'No': 0,
    'Yes': 1
})

# Split blood pressure into systolic & diastolic
bp_split = df['blood_pressure'].str.split('/', expand=True)
df['bp_systolic'] = bp_split[0].astype(int)
df['bp_diastolic'] = bp_split[1].astype(int)

df.drop(columns=['blood_pressure', 'patient_id'], inplace=True)

# =====================================================
# 3. FEATURE SELECTION
# =====================================================
features = [
    'age',
    'gender',
    'cholesterol',
    'bmi',
    'diabetes',
    'hypertension',
    'medication_count',
    'length_of_stay',
    'discharge_destination',
    'bp_systolic',
    'bp_diastolic'
]

X = df[features]
y = df['readmitted_30_days']

# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    random_state=42,
)

# =====================================================
# 5. PREPROCESSING
# =====================================================
numeric_features = [
    'age', 'cholesterol', 'bmi', 'medication_count',
    'length_of_stay', 'bp_systolic', 'bp_diastolic'
]

categorical_features = [
    'gender', 'diabetes', 'hypertension', 'discharge_destination'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# =====================================================
# 6. MODEL PIPELINE
# =====================================================
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# =====================================================
# 7. TRAIN MODEL
# =====================================================
model_pipeline.fit(X_train, y_train)

# ======================================================
print("Train:", model_pipeline.score(X_train, y_train))
print("Test:", model_pipeline.score(X_test, y_test))

# =====================================================
# 8. EVALUATION
# =====================================================
y_pred = model_pipeline.predict(X_test)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred), 3)
print(classification_report(y_test, y_pred))

# =====================================================
# 9. RECOMMENDATION ENGINE
# =====================================================
def generate_recommendations(patient):
    recs = []

    if patient['length_of_stay'] > 5:
        recs.append("Schedule early post-discharge follow-up.")

    if patient['medication_count'] > 5:
        recs.append("Perform medication reconciliation.")

    if patient['diabetes'] == 'Yes':
        recs.append("Provide diabetes management counseling.")

    if patient['hypertension'] == 'Yes':
        recs.append("Monitor blood pressure regularly.")

    if patient['bmi'] > 30:
        recs.append("Recommend lifestyle and diet modification.")

    if not recs:
        recs.append("Standard discharge protocol recommended.")

    return recs

# =====================================================
# 10. PREDICTION FUNCTION
# =====================================================
def predict_readmission(patient_data):
    patient_df = pd.DataFrame([patient_data])

    prediction = model_pipeline.predict(patient_df)[0]
    probability = model_pipeline.predict_proba(patient_df)[0][1]

    return {
        "Readmission Risk": "High" if prediction == 1 else "Low",
        "Risk Probability": round(probability, 2),
        "Recommendations": generate_recommendations(patient_data)
    }

# =====================================================
# 11. SAMPLE TEST
# =====================================================
if __name__ == "__main__":
    sample_patient = {
        'age': 72,
        'gender': 'Female',
        'cholesterol': 250,
        'bmi': 33.5,
        'diabetes': 'Yes',
        'hypertension': 'Yes',
        'medication_count': 8,
        'length_of_stay': 7,
        'discharge_destination': 'Home',
        'bp_systolic': 140,
        'bp_diastolic': 90
    }

    result = predict_readmission(sample_patient)

    print("\n🏥 Prediction Output")
    for k, v in result.items():
        print(f"{k}: {v}")
