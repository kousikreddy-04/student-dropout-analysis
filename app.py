from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import joblib
import os
import requests
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
import io

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize Flask App
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Configuration ---
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:admin@localhost:5432/student_dropout_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}
db = SQLAlchemy(app)

# --- Gemini AI Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key="
API_KEY = os.environ.get("GEMINI_API_KEY")

# --- ML Model Loading ---
try:
    model_artifacts = joblib.load('model_assets/lgbm_model.pkl')
    model = model_artifacts['model']
    label_encoders = model_artifacts['label_encoders']
    status_encoder = model_artifacts['status_encoder']
    feature_names = model_artifacts['feature_names']
    print("✅ ML model and artifacts loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Model file 'model_assets/lgbm_model.pkl' not found.")
    model = None

# --- Database Model Definition ---
class StudentPrediction(db.Model):
    __tablename__ = 'student_predictions'
    id = db.Column(db.Integer, primary_key=True)
    school_name = db.Column(db.String(255))
    area_type = db.Column(db.String(50))
    gender = db.Column(db.String(20))
    caste = db.Column(db.String(50))
    standard = db.Column(db.Integer)
    age = db.Column(db.Integer)
    year = db.Column(db.Integer)
    district = db.Column(db.String(100))
    dropout_reason = db.Column(db.String(100))
    parental_education = db.Column(db.String(100))
    family_income = db.Column(db.BigInteger)
    prev_academic_performance = db.Column(db.Float)
    attendance_record = db.Column(db.Float)
    teacher_student_ratio = db.Column(db.Float)
    distance_km = db.Column(db.Float)
    predicted_dropout_status = db.Column(db.String(50))
    predicted_risk_score = db.Column(db.Float)
    # REMOVED: risk_level is no longer needed
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

# --- Routes ---
@app.route('/')
def home():
    try:
        # Using the database is more robust for dropdowns
        school_names_query = db.session.query(StudentPrediction.school_name).distinct().order_by(StudentPrediction.school_name)
        districts_query = db.session.query(StudentPrediction.district).distinct().order_by(StudentPrediction.district)
        reasons_query = db.session.query(StudentPrediction.dropout_reason).filter(StudentPrediction.dropout_reason.isnot(None), StudentPrediction.dropout_reason != 'Not applicable').distinct().order_by(StudentPrediction.dropout_reason)
        school_names = [item[0] for item in school_names_query if item[0]]
        districts = [item[0] for item in districts_query if item[0]]
        dropout_reasons = [item[0] for item in reasons_query if item[0]]
    except Exception as e:
        print(f"❌ DATABASE ERROR while populating dropdowns: {e}")
        school_names, districts, dropout_reasons = [], [], []
    return render_template('index.html', school_names=school_names, districts=districts, dropout_reasons=dropout_reasons)

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Machine learning model is not loaded.'}), 500
    try:
        data = request.json
        response_data, db_entry_data = process_single_prediction(data)
        new_entry = StudentPrediction(**db_entry_data)
        db.session.add(new_entry)
        db.session.commit()
        return jsonify(response_data)
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not model: return jsonify({'error': 'Machine learning model is not loaded.'}), 500
    if 'csv_file' not in request.files: return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['csv_file']
    if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
    try:
        csv_data = io.StringIO(file.stream.read().decode("UTF8"))
        original_df = pd.read_csv(csv_data)
        processing_df = original_df.copy()
        processing_df.columns = [col.strip().replace(' ', '_').lower() for col in processing_df.columns]
        predictions_list = []
        db_entries = []
        for index, row in processing_df.iterrows():
            student_data = row.to_dict()
            response_data, db_entry_data = process_single_prediction(student_data, with_ai=False)
            result_row = original_df.iloc[index].to_dict()
            result_row['Predicted Dropout Status'] = response_data['prediction']
            result_row['Predicted Risk Score (%)'] = response_data['risk_score']
            predictions_list.append(result_row)
            db_entries.append(StudentPrediction(**db_entry_data))
        db.session.bulk_save_objects(db_entries)
        db.session.commit()
        return jsonify({'message': f'Successfully processed {len(db_entries)} records.', 'predictions': predictions_list})
    except Exception as e:
        db.session.rollback()
        print(f"❌ Batch Prediction Error: {e}")
        return jsonify({'error': 'An error occurred during batch processing.'}), 500

@app.route('/api/kpi_data')
def get_kpi_data():
    """Provides live KPI data from the database."""
    try:
        total_students = db.session.query(StudentPrediction).count()
        total_dropout = db.session.query(StudentPrediction).filter_by(predicted_dropout_status='Dropout').count()
        total_enrolled = total_students - total_dropout
        dropout_rate = (total_dropout / total_students) if total_students > 0 else 0
        return jsonify({
            'total_students': total_students,
            'total_dropout': total_dropout,
            'total_enrolled': total_enrolled,
            'dropout_rate': dropout_rate
        })
    except Exception as e:
        print(f"🔴 KPI Error: {e}")
        return jsonify({'error': 'Could not fetch KPI data.'}), 500

# --- HELPER FUNCTIONS ---
def process_single_prediction(data, with_ai=True):
    numeric_fields = {'standard': int, 'age': int, 'year': 'Int64', 'family_income': 'Int64', 'prev_academic_performance': float, 'attendance_record': float, 'teacher_student_ratio': float, 'distance_km': float}
    model_input_df = pd.DataFrame([data])
    for col, dtype in numeric_fields.items():
        if col in model_input_df.columns:
            model_input_df[col] = pd.to_numeric(model_input_df[col], errors='coerce').astype(dtype)
    processed_df = model_input_df.copy()
    for col, le in label_encoders.items():
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if pd.notna(x) and x in le.classes_ else -1)
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
    prediction_encoded = model.predict(processed_df)[0]
    prediction_proba = model.predict_proba(processed_df)[0]
    predicted_status = status_encoder.inverse_transform([prediction_encoded])[0]
    dropout_class_index = status_encoder.classes_.tolist().index('Dropout')
    dropout_risk_prob = prediction_proba[dropout_class_index]
    risk_score = round(dropout_risk_prob * 100, 2)
    response_data = {'prediction': predicted_status, 'risk_score': risk_score, 'interventions': None}
    if with_ai and predicted_status == "Dropout":
        response_data['interventions'] = get_gemini_interventions(data, response_data)
    db_entry_data = data.copy()
    db_entry_data.pop('dropout_status', None)
    db_entry_data['predicted_dropout_status'] = predicted_status
    db_entry_data['predicted_risk_score'] = float(risk_score)
    return response_data, db_entry_data

def get_gemini_interventions(student_data, prediction_result):
    if not API_KEY: return "Gemini API Key not configured."
    try:
        # REMOVED: risk_level from the prompt for consistency
        prompt = f"A model predicts a student will '{prediction_result['prediction']}' with a risk score of {prediction_result['risk_score']}%. Student data: {student_data}. Provide 3-4 concise intervention strategies."
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{GEMINI_API_URL}{API_KEY}", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return "Could not retrieve AI interventions."

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

