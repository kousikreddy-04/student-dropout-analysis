-- This script creates the necessary tables for user authentication and data storage.
-- Run this once in your PostgreSQL database to set up the project.

-- Drop existing tables to start with a clean schema for a fresh installation
DROP TABLE IF EXISTS public.student_predictions;
DROP TABLE IF EXISTS public.users;

-- Create the 'users' table to store login and role information
CREATE TABLE IF NOT EXISTS public.users
(
    id SERIAL PRIMARY KEY,
    username VARCHAR(150) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('teacher', 'admin'))
);

-- Create the 'student_predictions' table with a foreign key to the 'users' table
CREATE TABLE IF NOT EXISTS public.student_predictions
(
    id SERIAL PRIMARY KEY,
    student_name character varying(255),
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
	risk_level character varying(50),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    user_id integer REFERENCES users(id) -- Foreign key to link each prediction to a user
);

-- Set table owners (optional, but good practice)
ALTER TABLE IF EXISTS public.users OWNER to student;
ALTER TABLE IF EXISTS public.student_predictions OWNER to student;

COMMENT ON TABLE public.users IS 'Stores user accounts and their roles (teacher or admin).';
COMMENT ON TABLE public.student_predictions IS 'Stores historical data, single predictions, and batch predictions.';
COMMENT ON COLUMN public.student_predictions.user_id IS 'Identifies which user created this prediction record.';

-- End of database setup script