import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("--- Starting Model Training ---")

# --- 1. Load Data ---
try:
    df = pd.read_csv('karnataka_dropout_balanced.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'karnataka_dropout_enhanced_with_family.csv' not found.")
    exit()

# --- 2. Preprocessing ---
# FIX: Standardize all column names to snake_case for consistency
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
print("✅ Column names standardized to snake_case.")

# Define features and target using the new snake_case names
categorical_features = ['area_type', 'gender', 'caste', 'district', 'parental_education']
numerical_features = ['standard', 'age', 'year', 'family_income', 'prev_academic_performance', 'attendance_record', 'teacher_student_ratio', 'distance_km']
target_column = 'dropout_status'

# Apply Label Encoding
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Categorical features encoded.")

status_encoder = LabelEncoder()
df[target_column] = status_encoder.fit_transform(df[target_column])
print(f"✅ Target variable '{target_column}' encoded. Classes: {status_encoder.classes_}")

# --- 3. Feature Selection & Data Splitting ---
features = categorical_features + numerical_features
X = df[features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data split into training and testing sets.")

# --- 4. Model Training ---
print("⏳ Training the LightGBM model...")
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 5. Model Evaluation ---
y_pred = lgbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=status_encoder.classes_))
print("------------------------\n")

# --- 6. Save Model and Artifacts ---
ASSETS_DIR = 'model_assets'
MODEL_PATH = os.path.join(ASSETS_DIR, 'lgbm_model.pkl')

if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
    print(f"✅ Created directory: '{ASSETS_DIR}'")

# All artifacts will now consistently use snake_case
model_artifacts = {
    'model': lgbm,
    'label_encoders': label_encoders,
    'status_encoder': status_encoder,
    'feature_names': features
}

joblib.dump(model_artifacts, MODEL_PATH)
print(f"✅ Model and artifacts successfully saved to '{MODEL_PATH}'")

