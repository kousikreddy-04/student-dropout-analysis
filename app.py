# --- Import Necessary Libraries ---
from flask import Flask, request, jsonify, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import os
import requests
import io
from fpdf import FPDF
import openpyxl
import datetime # Import datetime for timezone handling
from sqlalchemy.exc import IntegrityError # Kept for robust error handling
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Flask App Initialization and Configuration ---
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-super-secret-key-for-development')
CORS(app)

# --- Database Configuration ---
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:admin@localhost:5432/student_dropout_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}
db = SQLAlchemy(app)

# --- Login Manager Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)

# --- Gemini AI Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent?key="
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

# --- Database Model Definitions ---

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False) # 'teacher' or 'admin'
    
    predictions = db.relationship('StudentPrediction', backref='user', lazy='dynamic')


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class StudentPrediction(db.Model):
    __tablename__ = 'student_predictions'
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(255))
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
    risk_level = db.Column(db.String(50)) 
    created_at = db.Column(db.TIMESTAMP(timezone=True), server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('users.id')) 

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) if hasattr(db.session, 'get') else User.query.get(int(user_id))


@app.route('/')
def index():
    """Renders the main single-page application shell."""
    return render_template('index.html')

# --- AUTHENTICATION API ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username, password, role = data.get('username'), data.get('password'), data.get('role', 'teacher')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required.'}), 400
    if role not in ['teacher', 'admin']:
        role = 'teacher' 

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists.'}), 409
        
    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    try:
        db.session.commit()
        return jsonify({'success': True, 'message': f'Registration successful for role: {role}. Please log in.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    if user and user.check_password(data.get('password')):
        login_user(user)
        return jsonify({'success': True, 'user': {'username': user.username, 'role': user.role}})
    return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})

@app.route('/api/check_auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'user': {'username': current_user.username, 'role': current_user.role}})
    return jsonify({'authenticated': False})

# --- DATA & PREDICTION API ---

def get_base_query():
    """Returns a base query for student predictions, filtered by user role."""
    if current_user.is_authenticated and current_user.role == 'teacher':
        return StudentPrediction.query.filter_by(user_id=current_user.id)
    return StudentPrediction.query

@app.route('/api/public_kpi_data')
def get_public_kpi_data():
    """Provides high-level, anonymized data for the public-facing dashboard."""
    try:
        total_students = StudentPrediction.query.count()
        total_dropout = StudentPrediction.query.filter_by(predicted_dropout_status='Dropout').count()
        dropout_rate = (total_dropout / total_students * 100) if total_students > 0 else 0
        return jsonify({'total_students': total_students, 'dropout_rate': dropout_rate})
    except Exception as e:
        print(f"🔴 Public KPI Error: {e}")
        return jsonify({'total_students': 0, 'dropout_rate': 0})

@app.route('/api/kpi_data')
@login_required
def get_kpi_data():
    """Provides detailed KPI data for authenticated users based on their role."""
    try:
        query = get_base_query()
        total_students = query.count()
        total_dropout = query.filter_by(predicted_dropout_status='Dropout').count()
        total_enrolled = total_students - total_dropout
        
        # NOTE: high_risk_students count REMOVED as requested.
        
        dropout_rate = (total_dropout / total_students * 100) if total_students > 0 else 0
        return jsonify({
            'total_students': total_students, 
            'total_dropout': total_dropout, 
            'total_enrolled': total_enrolled, 
            'dropout_rate': dropout_rate
        })
    except Exception as e:
        print(f"❌ Authenticated KPI Error: {e}")
        return jsonify({'error': 'Could not fetch KPI data.'}), 500

@app.route('/api/dropdown_data')
@login_required 
def get_dropdown_data():
    """Provides data for form dropdowns and report filters from the database."""
    try:
        query = get_base_query()
        
        school_names = [item[0] for item in query.with_entities(StudentPrediction.school_name).distinct().order_by(StudentPrediction.school_name) if item[0]]
        districts = [item[0] for item in query.with_entities(StudentPrediction.district).distinct().order_by(StudentPrediction.district) if item[0]]
        reasons = [item[0] for item in query.with_entities(StudentPrediction.dropout_reason).filter(StudentPrediction.dropout_reason.isnot(None)).distinct().order_by(StudentPrediction.dropout_reason) if item[0]]
        
        # Collect years as integers for proper sorting
        years = sorted([item[0] for item in query.with_entities(StudentPrediction.year).distinct() if item[0]])
        
        # Convert years back to string for consistent JSON output (though Python handles int)
        years = [str(y) for y in years] 

        return jsonify({
            'schools': school_names, 
            'districts': districts, 
            'reasons': reasons,
            'years': years
        })
    except Exception as e:
        print(f"❌ Error fetching dropdown data: {e}")
        return jsonify({'schools': [], 'districts': [], 'reasons': [], 'years': []})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model: return jsonify({'error': 'Model not loaded.'}), 500
    try:
        data = request.json
        response_data, db_entry_data = process_single_prediction(data)
        
        db_entry_data['user_id'] = current_user.id 
        new_entry = StudentPrediction(**db_entry_data)
        
        db.session.add(new_entry)
        db.session.commit()
        
        return jsonify(response_data)
    except Exception as e:
        db.session.rollback()
        print(f"❌ Prediction Error: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

@app.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    if not model: return jsonify({'error': 'Model not loaded.'}), 500
    if 'csv_file' not in request.files: return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['csv_file']
    if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
    
    try:
        csv_data = io.StringIO(file.stream.read().decode("UTF8"))
        original_df = pd.read_csv(csv_data)
        processing_df = original_df.copy()
        processing_df.columns = [col.strip().replace(' ', '_').lower() for col in processing_df.columns]
        
        predictions_list, db_entries = [], []
        
        for index, row in processing_df.iterrows():
            student_data = row.to_dict()
            
            response_data, db_entry_data = process_single_prediction(student_data, with_ai=False)
            
            db_entry_data['user_id'] = current_user.id
            
            result_row = {key.replace('_', ' ').title(): value for key, value in original_df.iloc[index].to_dict().items()}
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
        return jsonify({'error': f'An error occurred during batch processing: {str(e)}'}), 500

@app.route('/api/generate_report', methods=['POST'])
@login_required
def generate_report():
    try:
        data = request.json
        filters = data.get('filters', {})
        action = data.get('action', 'preview')
        report_format = data.get('format', 'excel')
        
        query = get_base_query()
        
        # --- Apply Dynamic Filters ---
        report_type = filters.get('report_type')
        if report_type == 'enrolled':
            query = query.filter(StudentPrediction.predicted_dropout_status == 'Enrolled')
        elif report_type == 'dropout':
            query = query.filter(StudentPrediction.predicted_dropout_status == 'Dropout')
            
        if filters.get('years'):
            query = query.filter(StudentPrediction.year.in_([int(y) for y in filters['years']]))
        if filters.get('districts'):
            query = query.filter(StudentPrediction.district.in_(filters['districts']))
        if filters.get('schools'):
            query = query.filter(StudentPrediction.school_name.in_(filters['schools']))
        if filters.get('caste'):
            query = query.filter(StudentPrediction.caste.in_(filters['caste']))
        if filters.get('gender'):
            query = query.filter(StudentPrediction.gender.in_(filters['gender']))
        if filters.get('area_type'):
            query = query.filter(StudentPrediction.area_type.in_(filters['area_type']))
            
        # --- End Apply Dynamic Filters ---
            
        if action == 'preview':
            total_count = query.count()
            preview_results = query.order_by(StudentPrediction.id.desc()).limit(10).all()
            
            preview_data = []
            for row in preview_results:
                row_dict = {}
                for c in row.__table__.columns:
                    value = getattr(row, c.name)
                    # FIX 1: Convert Timestamps to timezone-naive string for preview
                    if c.type.python_type is datetime.datetime and value:
                        row_dict[c.name] = value.replace(tzinfo=None).isoformat()
                    else:
                        row_dict[c.name] = value
                preview_data.append(row_dict)

            return jsonify({'total_count': total_count, 'preview_data': preview_data})
        
        elif action == 'download':
            results = query.all()
            
            # --- FIX 2: Pre-process datetimes for EXCEL/PDF export ---
            data_list = []
            for row in results:
                row_dict = {}
                for c in row.__table__.columns:
                    value = getattr(row, c.name)
                    if c.type.python_type is datetime.datetime and value:
                        # Convert to timezone-naive string for safety in export
                        row_dict[c.name] = value.replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        row_dict[c.name] = value
                data_list.append(row_dict)
            
            results_df = pd.DataFrame(data_list)
            # -----------------------------------------------------------
            
            results_df.columns = [col.replace('_', ' ').title() for col in results_df.columns]
            
            buffer = io.BytesIO()
            
            if report_format == 'excel':
                results_df.to_excel(buffer, index=False, engine='openpyxl')
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                filename = 'student_report.xlsx'
                
            elif report_format == 'pdf':
                pdf = FPDF(orientation='L', unit='mm', format='A3')
                pdf.add_page()
                pdf.set_font('Arial', 'B', 8)
                
                report_columns = ['Student Name', 'School Name', 'District', 'Predicted Dropout Status', 'Predicted Risk Score', 'Risk Level', 'Created At']
                col_widths = {'Student Name': 40, 'School Name': 50, 'District': 30, 'Predicted Dropout Status': 25, 'Predicted Risk Score': 25, 'Risk Level': 20, 'Created At': 35}
                default_width = 20
                
                for header in report_columns:
                    width = col_widths.get(header, default_width)
                    pdf.cell(width, 10, header, 1, 0, 'C')
                pdf.ln()
                
                pdf.set_font('Arial', '', 7)
                for index, row in results_df.iterrows():
                    for header in report_columns:
                        width = col_widths.get(header, default_width)
                        cell_text = str(row.get(header, 'N/A'))
                        if header == 'Predicted Risk Score':
                            cell_text = f"{float(cell_text):.2f}" if cell_text != 'N/A' else 'N/A'

                        pdf.cell(width, 5, cell_text, 1, 0, 'L')
                    pdf.ln()

                pdf_output = pdf.output(dest='S').encode('latin-1')
                buffer.write(pdf_output)
                mimetype = 'application/pdf'
                filename = 'student_report.pdf'
                
            else:
                return jsonify({'error': 'Invalid report format.'}), 400
            
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=filename, mimetype=mimetype)
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

# --- HELPER FUNCTIONS ---
def process_single_prediction(data, with_ai=True):
    model_data = data.copy()
    expected_model_features = ['area_type', 'gender', 'caste', 'standard', 'age', 'year', 'district', 'parental_education', 'family_income', 'prev_academic_performance', 'attendance_record', 'teacher_student_ratio', 'distance_km']
    
    model_input_data = {k: model_data.get(k) for k in expected_model_features if k in expected_model_features}
    
    numeric_fields = {'standard': int, 'age': int, 'year': 'Int64', 'family_income': 'Int64', 'prev_academic_performance': float, 'attendance_record': float, 'teacher_student_ratio': float, 'distance_km': float}
    
    model_input_df = pd.DataFrame([model_input_data])
    
    for col, dtype in numeric_fields.items():
        if col in model_input_df.columns:
            model_input_df[col] = pd.to_numeric(model_input_df[col], errors='coerce').astype(dtype)
    
    processed_df = model_input_df.copy()
    
    for col, le in label_encoders.items():
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if pd.notna(x) and x in le.classes_ else -1)
            
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
    
    # --- Prediction ---
    prediction_encoded = model.predict(processed_df)[0]
    prediction_proba = model.predict_proba(processed_df)[0]
    
    predicted_status = status_encoder.inverse_transform([prediction_encoded])[0]
    dropout_class_index = status_encoder.classes_.tolist().index('Dropout')
    dropout_risk_prob = prediction_proba[dropout_class_index]
    risk_score = round(dropout_risk_prob * 100, 2)
    
    risk_level = "Low"
    if risk_score >= 70: risk_level = "High"
    elif risk_score >= 40: risk_level = "Medium"
    
    # --- Response Data ---
    response_data = {'prediction': predicted_status, 'risk_level': risk_level, 'risk_score': risk_score, 'interventions': None}
    
    if with_ai and risk_level in ["High", "Medium"]:
        response_data['interventions'] = get_gemini_interventions(data, response_data)
        
    # --- DB Entry Data ---
    db_entry_data = data.copy()
    db_entry_data.pop('dropout_status', None) 
    
    db_entry_data['predicted_dropout_status'] = predicted_status
    db_entry_data['predicted_risk_score'] = float(risk_score)
    db_entry_data['risk_level'] = risk_level
    
    return response_data, db_entry_data

def get_gemini_interventions(student_data, prediction_result):
    if not API_KEY: return "Gemini API Key not configured."
    try:
        prompt = f"A model predicts a student named {student_data.get('student_name', 'Unknown')} will '{prediction_result['prediction']}' with a risk level of '{prediction_result['risk_level']}' ({prediction_result['risk_score']}% risk score). Student data: {student_data}. Provide 3-4 concise intervention strategies."
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(f"{GEMINI_API_URL}{API_KEY}", json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return "Could not retrieve AI interventions."

if __name__ == '__main__':
    with app.app_context():
        # NOTE: db.create_all() relies on the database.sql having been run 
        # successfully once to create the tables with the correct schema.
        db.create_all() 
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)