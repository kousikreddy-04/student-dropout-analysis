import pandas as pd
from sqlalchemy import create_engine
import sys

def load_historical_data():
    """
    Reads historical data from the CSV file, formats it to match the
    'student_predictions' table schema, and loads it into the PostgreSQL database.
    This is intended as a one-time operation to seed the database with historical records.
    """
    # --- Configuration ---
    # IMPORTANT: UPDATE THIS LINE with the same PostgreSQL credentials used in your app.py
    DATABASE_URI = 'postgresql://postgres:kousik@localhost:5432/student'
    CSV_FILE_PATH = 'karnataka_dropout_enhanced_with_family.csv'
    TABLE_NAME = 'student_predictions'

    # --- 1. Connect to the Database ---
    try:
        engine = create_engine(DATABASE_URI)
        print("Successfully connected to the PostgreSQL database.")
    except Exception as e:
        print(f"Error: Could not connect to the database. Ensure PostgreSQL is running and credentials are correct.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 2. Load Data from CSV ---
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"'{CSV_FILE_PATH}' loaded successfully. Found {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found in the project directory.")
        sys.exit(1)

    # --- 3. Prepare the DataFrame to match the DB schema ---
    
    # Define a mapping from CSV column names to database column names
    column_mapping = {
        'School Name': 'school_name',
        'Area Type': 'area_type',
        'Gender': 'gender',
        'Caste': 'caste',
        'Standard': 'standard',
        'Age': 'age',
        'Year': 'year',
        'District': 'district',
        'Dropout Reason': 'dropout_reason',
        'Parental_Education': 'parental_education',
        'Family_Income': 'family_income',
        'Prev_Academic_Performance': 'prev_academic_performance',
        'Attendance_Record': 'attendance_record',
        'Teacher_Student_Ratio': 'teacher_student_ratio',
        'Distance_km': 'distance_km',
        'Dropout Status': 'predicted_dropout_status'
    }

    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)

    # Add the 'predicted_risk_score' column, which doesn't exist in the historical data.
    # It will be stored as NULL in the database.
    df['predicted_risk_score'] = None

    # Ensure the final DataFrame only contains columns that exist in the database table
    final_columns = list(column_mapping.values()) + ['predicted_risk_score']
    df_final = df[final_columns]
    
    print("Data prepared for database insertion.")
    
    # --- 4. Load data into the PostgreSQL table ---
    try:
        # Use if_exists='append' to add the data without deleting existing records.
        # Use index=False to avoid writing the pandas DataFrame index to the DB.
        df_final.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
        print(f"\nSuccess! Loaded {len(df_final)} rows of historical data into the '{TABLE_NAME}' table.")
        print("You can now run 'flask run' to start the web application.")
    except Exception as e:
        print(f"\nError: Failed to load data into the database.")
        print("Possible reason: The table schema in 'database_setup.sql' might not match the CSV columns.")
        print(f"Details: {e}")

if __name__ == '__main__':
    print("Starting historical data loading process...")
    print("Important: Ensure you have run the 'database_setup.sql' script first to create a clean table.")
    load_historical_data()

