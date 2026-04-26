import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier


df = pd.read_csv("hospital_readmissions_30k.csv")


df['readmitted_30_days'] = df['readmitted_30_days'].map({
    'No': 0,
    'Yes': 1
})
# Handle missing values
bp = df['blood_pressure'].str.split('/', expand=True)
df['bp_systolic'] = bp[0].astype(int)
df['bp_diastolic'] = bp[1].astype(int)

df.drop(columns=['blood_pressure', 'patient_id'], inplace=True)

# Features and Target
X = df.drop(columns=['readmitted_30_days'])
y = df['readmitted_30_days']


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_col = X_train.select_dtypes(include='number').columns.tolist()
cat_col = X_train.select_dtypes(include='object').columns.tolist() 

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_col),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
])


# XGBoost Model
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Scores
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)


print("\n📈 Model Scores")
print(f"Train Accuracy : {train_accuracy}")
print(f"Test Accuracy  : {test_accuracy}")


model_path="Hospital_model.pkl"
joblib.dump(model,model_path)
print(f"\nModel saved to {model_path}")