# 🎓 Student Dropout Prediction & Intervention System

A full-stack web application that leverages a machine learning model to predict student dropout risk in real-time.  
The system features an **interactive dashboard** with live KPIs, an **AI-powered intervention suggestion engine** using the **Google Gemini API**, and **batch prediction** capabilities for processing large datasets.

🔗 **Live Demo**: [https://student-dropout-analysis-1.onrender.com/](https://student-dropout-analysis-1.onrender.com/)  


---

## 🚀 Key Features
- **Real-Time Prediction**: Input a single student’s data and instantly get a dropout risk score.  
- **AI-Powered Interventions**: For at-risk students, the system calls **Google Gemini API** to generate actionable strategies for educators.  
- **Live KPI Dashboard**: View real-time metrics like *Total Students* and *Dropout Rate*.  
- **Batch Prediction**: Upload a CSV file with multiple student records, process them in batch, and download results.  
- **Embedded Business Intelligence**: Integrates with **Power BI** for detailed, interactive reports.  
- **Scalable Architecture**: Built with Flask + PostgreSQL, deployable on cloud platforms like **Render**.  

---

## 🏗️ Project Architecture
The system follows a decoupled, scalable architecture:

- **Frontend (UI):**  
  HTML, Tailwind CSS, and JavaScript (single-page app).  
- **Backend (API Server):**  
  Flask app exposing endpoints:  
  - `/predict` → Single prediction  
  - `/batch_predict` → Batch CSV processing  
  - `/api/kpi_data` → Live KPI data for dashboard  
- **Machine Learning Model:**  
  LightGBM (LGBM) Classifier trained on historical data (`.pkl` file).  
- **Database:**  
  PostgreSQL for storing predictions and historical records.  
- **AI Integration:**  
  Google Gemini API for tailored intervention strategies.  
- **Business Intelligence:**  
  Power BI connected to the database via **DirectQuery** (live updates).  

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Gunicorn  
- **Frontend:** HTML, Tailwind CSS, JavaScript  
- **Database:** PostgreSQL  
- **Machine Learning:** Scikit-learn, LightGBM, Pandas  
- **AI:** Google Gemini API  
- **Visualization:** Power BI  
- **Deployment:** Render, Git, Git LFS (for `.pkl` model)  

---

## ⚙️ Local Setup and Installation

### 📋 Prerequisites
- Python 3.10+  
- PostgreSQL installed & running  
- Git + Git LFS  

### 🖥️ Step-by-Step Guide
1. **Clone the Repository**
   ```bash
   git clone https://github.com/kousikreddy-04/student-dropout-analysis
   cd your-repo-name
   ```

2. **Install Git LFS**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**  
   Create a `.env` file in the project root (see `.env.example`):
   ```ini
   DATABASE_URL="postgresql://postgres:your_password@localhost:5432/student_dropout_db"
   GEMINI_API_KEY="your_google_ai_api_key"
   ```

6. **Create the Database**  
   Use pgAdmin or psql to create the DB (`student_dropout_db`).

7. **Train ML Model**
   ```bash
   python ml_model.py
   ```

8. **Load Historical Data**
   ```bash
   python load_historical_data.py
   ```

9. **Run Flask App**
   ```bash
   flask run
   ```
   Now open → [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ☁️ Deployment on Render
1. **Push to GitHub** (make sure `.gitattributes` includes Git LFS).  
2. **Create PostgreSQL Service** on Render.  
3. **Create Web Service** and connect GitHub repo.  
   - **Build Command:** `pip install -r requirements.txt`  
   - **Start Command:** `gunicorn app:app`  
4. **Add Environment Variables** in Render:
   - `DATABASE_URL` → Internal connection string  
   - `GEMINI_API_KEY` → Your Google Gemini API key  
5. **Seed the Database:**  
   Run `load_historical_data.py` locally once, using Render’s **external DB URL**.  

---

## 📊 Business Intelligence
Power BI connects directly to the live PostgreSQL DB using **DirectQuery mode**, ensuring dashboards are always up-to-date.

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
