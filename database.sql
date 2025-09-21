-- This script creates the table for storing student data and predictions.
-- Run this in your PostgreSQL database (e.g., using psql or a GUI like DBeaver/pgAdmin).

-- Drop the table if it already exists to start fresh (optional)
-- DROP TABLE IF EXISTS student_data;

-- Create the table
CREATE TABLE student_data (
    id SERIAL PRIMARY KEY,
    area_type VARCHAR(50),
    gender VARCHAR(50),
    caste VARCHAR(50),
    standard INT,
    age INT,
    year INT,
    district VARCHAR(100),
    parental_education VARCHAR(100),
    family_income INT,
    prev_academic_performance FLOAT,
    attendance_record FLOAT,
    teacher_student_ratio FLOAT,
    distance_km FLOAT,
    predicted_status VARCHAR(50),
    prediction_risk_percent FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optional: Add comments to columns for clarity
COMMENT ON COLUMN student_data.id IS 'Unique identifier for each record';
COMMENT ON COLUMN student_data.predicted_status IS 'The status predicted by the ML model (Dropout/Enrolled)';
COMMENT ON COLUMN student_data.prediction_risk_percent IS 'The model''s confidence in the dropout prediction (0-100)';
COMMENT ON COLUMN student_data.created_at IS 'Timestamp when the record was created';

-- You can insert some dummy data to test your Power BI connection if you wish
/*
INSERT INTO student_data (
    area_type, gender, caste, standard, age, year, district,
    parental_education, family_income, prev_academic_performance,
    attendance_record, teacher_student_ratio, distance_km,
    predicted_status, prediction_risk_percent
) VALUES (
    'Urban', 'Female', 'General', 8, 13, 2024, 'Bengaluru Urban',
    'Graduate', 75000, 85.5, 92.3, 25.5, 2.1,
    'Enrolled', 15.2
);
*/

-- Grant usage for your app's database user if necessary
-- Make sure the user you connect with in the Flask app has the correct permissions.
-- GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE student_data TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE student_data_id_seq TO your_app_user;
