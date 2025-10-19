# 🎓 Student Dropout Prediction & Intervention System

[![Built with Flask](https://img.shields.io/badge/Web%20Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![AI Powered by](https://img.shields.io/badge/AI-Google%20Gemini%20API-4285F4.svg)](https://ai.google.dev/gemini-api)

An **AI-powered web application** that predicts student dropout risk and assists educators with insights and interventions. Built using **Flask**, **PostgreSQL**, and **LightGBM**, it integrates with the **Google Gemini API** to generate personalized recommendations and uses **Power BI** for visualization.

🔗 **Live Demo:** [https://student-dropout-analysis-1.onrender.com/](https://student-dropout-analysis-1.onrender.com/)

---

## 🚀 Key Features

* Predict dropout probability **in real-time**.
* Auto-generates **AI-guided interventions** via **Google Gemini API**.
* Secure **Flask-Login based authentication** for teachers and administrators.
* **Batch uploads** for multiple students via CSV/Excel.
* Export reports as **PDF/Excel**.
* **Color-coded risk levels:** Red = High, Yellow = Medium, Green = Low.
* Cloud-ready deployment with **Render** and PostgreSQL integration.
* Supports **Power BI** dashboards for performance tracking.

---

## 🏗️ Project Architecture

The system follows a modular, scalable architecture with clearly defined layers:

* **Frontend (UI):** Developed using HTML, Tailwind CSS, and JavaScript. The interface interacts with Flask APIs for real-time prediction and visualization.
* **Backend (API Server):** Flask app providing REST endpoints for prediction, batch processing, dynamic KPIs, authentication, and report generation.
* **Machine Learning Model:** LightGBM classifier trained on socio-economic and academic datasets to predict dropout risk. Serialized model and encoders are loaded from `model_assets/lgbm_model.pkl`.
* **Database:** PostgreSQL using SQLAlchemy ORM with two core tables: `users` (credentials) and `student_predictions` (predictions, risk levels, and metadata).
* **AI Integration:** **Google Gemini API** generates human-interpretable interventions based on risk classification.
* **Business Intelligence Dashboard:** Power BI connected via **DirectQuery** for live visual analytics (school-wise, gender-wise, district-wise trends).

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | HTML, Tailwind CSS, JavaScript |
| **Backend** | Flask, Gunicorn |
| **Machine Learning** | scikit-learn, LightGBM, pandas, joblib |
| **Database** | PostgreSQL, SQLAlchemy |
| **AI** | **Google Gemini API** |
| **Visualization** | Power BI|
| **Deployment** | Render Cloud Platform |
| **File Handling** | openpyxl, FPDF |

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.10+
* PostgreSQL database
* Git and Git LFS installed
* Power BI Desktop (for dashboard setup)

### Steps

1.  Clone the repository:
    ```bash
    git clone [https://github.com/kousikreddy-04/student-dropout-analysis.git](https://github.com/kousikreddy-04/student-dropout-analysis.git)
    cd student-dropout-analysis
    ```
2.  Install Git LFS and pull large files:
    ```bash
    git lfs install
    git lfs pull
    ```
3.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/Mac
    venv\Scripts\activate # Windows
    ```
4.  Install project dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Configure environment variables (create a `.env` file):
    ```bash
    DATABASE_URL="postgresql://postgres:your_password@localhost:5432/student_dropout_db"
    GEMINI_API_KEY="your_google_ai_api_key"
    SECRET_KEY="your_secret_key"
    ```
6.  Initialize database schema (create tables):
    ```bash
    python
    from app import db
    db.create_all()
    exit()
    ```
7.  Train or verify model in `ml_model.py`, or ensure model file exists at `/model_assets/lgbm_model.pkl`.
8.  Run the Flask server:
    ```bash
    flask run
    ```
    Access via [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ☁️ Deployment on Render

1.  Push code to GitHub ensuring `.gitattributes` tracks LFS model files.
2.  Create a **PostgreSQL** instance and **Web Service** on Render.
3.  Set environment variables in Render Dashboard:
    * `DATABASE_URL`
    * `GEMINI_API_KEY`
    * `SECRET_KEY`
4.  Render build and start commands:
    ```bash
    Build: pip install -r requirements.txt
    Start: gunicorn app:app
    ```
5.  Optionally seed database locally via `load_historical_data.py` using the Render external DB URL.

---

## 📊 Business Intelligence

The Power BI dashboard connects directly to PostgreSQL via **DirectQuery** mode, displaying:

* School-wise dropout distribution
* Gender and caste-based risk trends
* KPI cards for enrolled vs predicted dropouts

---

## 📜 License

Licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

---

## 🌟 Acknowledgements

* **Google Gemini API** for generating insights
* Render for hosting services
* Power BI for interactive analytics
