import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. Load Data ---
try:
    df = pd.read_csv('karnataka_dropout_enhanced_with_family.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'karnataka_dropout_enhanced_with_family.csv' not found. Please make sure the dataset is in the same directory.")
    exit()

# --- 2. Preprocessing ---
# Define features and target based on the dataset
categorical_features = ['Area Type', 'Gender', 'Caste', 'District', 'Parental_Education']
numerical_features = ['Standard', 'Age', 'Year', 'Family_Income', 'Prev_Academic_Performance', 'Attendance_Record', 'Teacher_Student_Ratio', 'Distance_km']
target_column = 'Dropout Status'

# Apply Label Encoding to categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Categorical features encoded.")

# Encode the target variable ('Dropout' and 'Enrolled')
status_encoder = LabelEncoder()
df[target_column] = status_encoder.fit_transform(df[target_column])
print(f"✅ Target variable '{target_column}' encoded. Classes: {status_encoder.classes_}")

# --- 3. Feature Selection & Data Splitting ---
features = categorical_features + numerical_features
X = df[features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

# --- 4. Model Training ---
print("⏳ Training the LightGBM model...")
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 5. Model Evaluation ---
y_pred = lgbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy on test set: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=status_encoder.classes_))
print("------------------------\n")

# --- 6. Save Model and Artifacts ---
# Define the directory and path for the model assets
ASSETS_DIR = 'model_assets'
MODEL_PATH = os.path.join(ASSETS_DIR, 'lgbm_model.pkl')

# Create the 'model_assets' directory if it doesn't exist
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
    print(f"✅ Created directory: '{ASSETS_DIR}'")

# Bundle all artifacts (model, encoders, feature names) into a single dictionary
model_artifacts = {
    'model': lgbm,
    'label_encoders': label_encoders,
    'status_encoder': status_encoder,
    'feature_names': features
}

# Save the bundled artifacts to the specified path
joblib.dump(model_artifacts, MODEL_PATH)

print(f"✅ Model and all artifacts successfully saved to '{MODEL_PATH}'")

