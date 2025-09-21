import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    """
    Trains a LightGBM model on the student dropout dataset and saves it.
    """
    # 1. Load Data
    try:
        df = pd.read_csv('karnataka_dropout_enhanced_with_family.csv')
    except FileNotFoundError:
        print("Error: 'karnataka_dropout_enhanced_with_family.csv' not found.")
        print("Please place the dataset in the same directory as this script.")
        return

    print("Dataset loaded successfully.")

    # 2. Preprocessing
    # Drop non-essential columns for this model
    df = df.drop(['School Name', 'Dropout Reason'], axis=1)

    # Encode the target variable
    le_status = LabelEncoder()
    df['Dropout Status'] = le_status.fit_transform(df['Dropout Status'])
    # 'Dropout' will be 0, 'Enrolled' will be 1. We want to predict dropout.

    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Apply Label Encoding to all categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    print("Data preprocessing and encoding complete.")

    # 3. Define Features (X) and Target (y)
    X = df.drop('Dropout Status', axis=1)
    y = df['Dropout Status']

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Train LightGBM Model
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgbm.fit(X_train, y_train)

    print("Model training complete.")

    # 6. Evaluate Model
    y_pred = lgbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # We use inverse_transform to show original labels in the report
    print(classification_report(y_test, y_pred, target_names=le_status.inverse_transform([0, 1])))

    # 7. Save the Model and Encoders
    artifacts = {
        'model': lgbm,
        'label_encoders': label_encoders,
        'status_encoder': le_status,
        'feature_names': list(X.columns)
    }
    joblib.dump(artifacts, 'lgbm_dropout_model.joblib')
    print("\nTrained model and encoders have been saved to 'lgbm_dropout_model.joblib'")

if __name__ == '__main__':
    train_model()
