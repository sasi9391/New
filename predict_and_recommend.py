import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/readmission_model.pkl")

def generate_recommendations(patient):
    recommendations = []

    if patient['time_in_hospital'] > 7:
        recommendations.append("Schedule early post-discharge follow-up.")

    if patient['num_medications'] > 10:
        recommendations.append("Perform medication reconciliation.")

    if patient['number_emergency'] > 1:
        recommendations.append("Provide care coordination and ER avoidance plan.")

    if patient['number_inpatient'] > 1:
        recommendations.append("Enroll patient in chronic disease management program.")

    if not recommendations:
        recommendations.append("Standard discharge protocol recommended.")

    return recommendations


def predict_readmission(patient_data):
    df = pd.DataFrame([patient_data])

    risk = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    recommendations = generate_recommendations(patient_data)

    return {
        "readmission_risk": "High" if risk == 1 else "Low",
        "risk_probability": round(probability, 2),
        "recommendations": recommendations
    }


# 🔹 Example usage
if __name__ == "__main__":
    patient = {
        'age': '[60-70)',
        'time_in_hospital': 9,
        'num_lab_procedures': 55,
        'num_medications': 14,
        'number_outpatient': 0,
        'number_emergency': 2,
        'number_inpatient': 1,
        'gender': 'Female',
        'race': 'Caucasian'
    }

    result = predict_readmission(patient)

    print("\n🏥 Readmission Prediction Result")
    print(result)
