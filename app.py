from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import joblib
import os
import requests
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize Flask App
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Configuration ---
# THIS IS THE FIX: Securely load the database URL from Render's environment variables.
# It will fall back to a default localhost URL if the variable is not set (for local use).
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:admin@localhost:5432/student_dropout_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db = SQLAlchemy(app)

# --- Gemini AI Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key="
API_KEY = os.environ.get("GEMINI_API_KEY")

# --- ML Model Loading ---
try:
    # Ensure this path is correct for your project structure
    model_artifacts = joblib.load('model_assets/lgbm_model.pkl')
    model = model_artifacts['model']
    label_encoders = model_artifacts['label_encoders']
    status_encoder = model_artifacts['status_encoder']
    feature_names = model_artifacts['feature_names']
    print("✅ ML model and artifacts loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Model file not found. Please check the file name and path.")
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
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

# --- Routes ---
@app.route('/')
def home():
    try:
        df = pd.read_csv('karnataka_dropout_enhanced_with_family.csv')
        school_names = sorted(df['School Name'].unique().tolist())
        districts = sorted(df['District'].unique().tolist())
        dropout_reasons = sorted(df[df['Dropout Reason'] != 'Not applicable']['Dropout Reason'].unique().tolist())
    except FileNotFoundError:
        school_names, districts, dropout_reasons = [], [], []
    return render_template('index.html', 
                           school_names=school_names, 
                           districts=districts, 
                           dropout_reasons=dropout_reasons)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500
    try:
        data = request.json
        numeric_fields = {'standard': int, 'age': int, 'year': int, 'family_income': int, 'prev_academic_performance': float, 'attendance_record': float, 'teacher_student_ratio': float, 'distance_km': float}
        for field, func in numeric_fields.items():
            if field in data and data[field] not in [None, '']: data[field] = func(data[field])
        model_input_df = pd.DataFrame([data])
        processed_df = model_input_df.copy()
        for col, le in label_encoders.items():
            if col in processed_df.columns: processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
        prediction_encoded = model.predict(processed_df)[0]
        prediction_proba = model.predict_proba(processed_df)[0]
        predicted_status = status_encoder.inverse_transform([prediction_encoded])[0]
        dropout_class_index = status_encoder.classes_.tolist().index('Dropout')
        dropout_risk_prob = prediction_proba[dropout_class_index]
        risk_score = round(dropout_risk_prob * 100, 2)
        response_data = {'prediction': predicted_status, 'risk_score': risk_score, 'interventions': None}
        if predicted_status == "Dropout": response_data['interventions'] = get_gemini_interventions(data, response_data)
        new_entry = StudentPrediction(school_name=data.get('school_name'), area_type=data.get('area_type'), gender=data.get('gender'), caste=data.get('caste'), standard=data.get('standard'), age=data.get('age'), year=data.get('year'), district=data.get('district'), dropout_reason=data.get('dropout_reason'), parental_education=data.get('parental_education'), family_income=data.get('family_income'), prev_academic_performance=data.get('prev_academic_performance'), attendance_record=data.get('attendance_record'), teacher_student_ratio=data.get('teacher_student_ratio'), distance_km=data.get('distance_km'), predicted_dropout_status=predicted_status, predicted_risk_score=float(risk_score))
        db.session.add(new_entry)
        db.session.commit()
        return jsonify(response_data)
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'This exact student record already exists.'}), 409
    except Exception as e:
        db.session.rollback()
        print(f"❌ Prediction Error: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/kpi_data')
def get_kpi_data():
    try:
        total_students = db.session.query(StudentPrediction).count()
        total_dropout = db.session.query(StudentPrediction).filter_by(predicted_dropout_status='Dropout').count()
        total_enrolled = db.session.query(StudentPrediction).filter_by(predicted_dropout_status='Enrolled').count()
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

def get_gemini_interventions(student_data, prediction_result):
    if not API_KEY: return "Gemini API Key not configured."
    try:
        prompt = f"""
        You are an expert educational counselor in Karnataka, India. A machine learning model has identified a student as likely to '{prediction_result['prediction']}' with a risk score of {prediction_result['risk_score']}%.

        Student Profile:
        - Standard: {student_data.get('standard')}
        - Age: {student_data.get('age')}
        - Gender: {student_data.get('gender')}
        - District: {student_data.get('district')}
        - Previous Academic Performance: {student_data.get('prev_academic_performance')}%
        - Attendance Record: {student_data.get('attendance_record')}%
        - Anticipated Dropout Reason: {student_data.get('dropout_reason')}

        Provide 3-4 concise, actionable, and empathetic intervention strategies for a teacher.
        Format as a simple bulleted list using '-'. Do not add headers or introductory text.
        """
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{GEMINI_API_URL}{API_KEY}", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return "Could not retrieve AI interventions."

if __name__ == '__main__':
    # This block is for local development only and will not be run on Render.
    # The 'db.create_all()' line can be used locally to set up your database tables.
    with app.app_context():
        db.create_all()
        
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

