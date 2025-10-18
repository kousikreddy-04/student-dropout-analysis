import pandas as pd
from sqlalchemy import create_engine
import sys
from dotenv import load_dotenv
import os

def load_historical_data():
    """
    Connects to the PostgreSQL database, loads the new historical dataset,
    formats it, and appends it to the 'student_predictions' table.
    """
    load_dotenv()
    DATABASE_URI = os.environ.get('DATABASE_URL')
    # UPDATED: Using the new dataset file
    CSV_FILE_PATH = 'karnataka_dropout_balanced.csv'
    TABLE_NAME = 'student_predictions'

    if not DATABASE_URI:
        print("Error: DATABASE_URL not found in .env file.")
        sys.exit(1)
    try:
        engine = create_engine(DATABASE_URI)
        print("Successfully connected to the PostgreSQL database.")
    except Exception as e:
        print(f"Error: Could not connect to the database. Details: {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"'{CSV_FILE_PATH}' loaded successfully. Found {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        sys.exit(1)

    # Standardize column names to snake_case
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    print("✅ Column names standardized to snake_case.")

    # Create prediction columns from historical status
    if 'dropout_status' in df.columns:
        df['predicted_dropout_status'] = df['dropout_status']
        df['predicted_risk_score'] = df['dropout_status'].apply(lambda x: 100.0 if x == 'Dropout' else 0.0)
    else:
        print("Error: 'dropout_status' column not found in the CSV.")
        sys.exit(1)
        
    # Define the final columns for the database table, now including the real student_name
    final_columns = [
        'student_name',
        'school_name', 'area_type', 'gender', 'caste', 'standard', 'age', 'year',
        'district', 'dropout_reason', 'parental_education', 'family_income',
        'prev_academic_performance', 'attendance_record', 'teacher_student_ratio',
        'distance_km', 'predicted_dropout_status', 'predicted_risk_score'
    ]
    
    if not all(col in df.columns for col in final_columns):
        missing = [col for col in final_columns if col not in df.columns]
        print(f"Error: DataFrame is missing expected columns: {missing}")
        sys.exit(1)

    df_final = df[final_columns]
    
    print("Data prepared for database insertion.")
    
    try:
        # Use if_exists='append' to add data to the existing table after running setup.
        df_final.to_sql(TABLE_NAME, engine, if_exists='append', index=False, chunksize=1000)
        print(f"\nSuccess! Loaded {len(df_final)} rows into the '{TABLE_NAME}' table.")
        print("You can now run 'flask run' to start the web application.")
    except Exception as e:
        print(f"\nError: Failed to load data into the database. Details: {e}")

if __name__ == '__main__':
    print("Starting historical data loading process...")
    print("Important: Ensure you have run the 'database_setup.sql' script first.")
    load_historical_data()

