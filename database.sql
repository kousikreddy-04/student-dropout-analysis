-- This script creates the definitive table structure for the application.
-- Run this once in your PostgreSQL database to set up the table correctly.

-- Drop the table if it exists to ensure a clean start
DROP TABLE IF EXISTS public.student_predictions;

-- Create the table with an 'id' primary key and all columns expected by the app
CREATE TABLE IF NOT EXISTS public.student_predictions
(
    id SERIAL PRIMARY KEY,
    school_name character varying(255),
    area_type character varying(50),
    gender character varying(20),
    caste character varying(50),
    standard integer,
    age integer,
    year integer,
    district character varying(100),
    dropout_reason character varying(100),
    parental_education character varying(100),
    family_income bigint,
    prev_academic_performance double precision,
    attendance_record double precision,
    teacher_student_ratio double precision,
    distance_km double precision,
    predicted_dropout_status character varying(50),
    predicted_risk_score double precision,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);

-- Set the owner of the table
ALTER TABLE IF EXISTS public.student_predictions
    OWNER to student;

-- Add comments for clarity
COMMENT ON TABLE public.student_predictions
    IS 'Stores historical data, single predictions, and batch predictions.';

