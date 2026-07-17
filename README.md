# 🚀 DataMind AI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-yellow)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-orange)
![CSS3](https://img.shields.io/badge/CSS3-Styling-blue)
![License](https://img.shields.io/badge/License-MIT-red)

An AI-powered full-stack analytics platform that enables users to upload datasets, connect databases, query data using natural language, generate AI-powered insights, visualize analytics, and explore structured data through an intuitive web interface.

---

# 🌐 Live Links

### Frontend
https://data-mind-ai-beryl.vercel.app/

### Backend API
https://datamind-ai-0jk7.onrender.com

### API Documentation
https://datamind-ai-0jk7.onrender.com/docs

---

# ✨ Features

- 📂 Upload CSV and Excel datasets
- 🤖 AI-powered natural language querying
- 📊 Automated dataset profiling
- 📈 Interactive charts and visualizations
- 🧠 AI-generated executive summaries
- 📑 Schema detection and metadata extraction
- 🗄 MySQL database integration
- 📜 Query history
- ⚡ REST APIs built using FastAPI
- 🎨 Responsive modern dashboard

---

# 🏗 Architecture

```
                Frontend
       (HTML • CSS • JavaScript)
                     │
                     ▼
             FastAPI Backend
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 Dataset Upload   AI Processing   Database
    (CSV)        (OpenAI APIs)   Integration
     │               │               │
     └───────────────┼───────────────┘
                     ▼
      Charts • Insights • Summaries
```

---

# 🛠 Tech Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- FastAPI
- Pandas
- OpenAI API
- MySQL Connector
- Matplotlib

## Deployment

- Vercel
- Render

---

# 📡 REST APIs

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/csv/upload` | Upload datasets |
| GET | `/api/csv/files` | List uploaded datasets |
| GET | `/api/csv/{id}/read` | Read dataset |
| GET | `/api/csv/{id}/meta` | Dataset metadata |
| GET | `/api/csv/{id}/schema` | AI schema generation |
| POST | `/api/ai/query` | Natural language queries |
| POST | `/api/database/tables` | Connect database |

---

# 📂 Project Structure

```
DataMind-AI
│
├── frontend/
│
├── backend/
│   ├── routes/
│   ├── services/
│   ├── uploads/
│   ├── main.py
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

---

# 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/bhumika0810/DataMind-AI.git
```

Move into the project

```bash
cd DataMind-AI/backend
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the backend

```bash
uvicorn main:app --reload
```

Open the frontend

```
frontend/index.html
```

---

# 📸 Screenshots

(Add your screenshots here)

---

# 📌 Skills Demonstrated

- Full Stack Development
- REST API Development
- AI Integration
- FastAPI
- Data Analytics
- Data Visualization
- Database Connectivity
- Frontend Development
- Backend Development
- API Deployment

---

# ⚠ Notes

- Backend is deployed on Render.
- Frontend is deployed on Vercel.
- Database connectivity requires an accessible MySQL server (local or cloud-hosted) for successful connection.

---

# 🔮 Future Enhancements

- User Authentication
- Team Collaboration
- PostgreSQL Support
- MongoDB Support
- Cloud Storage Integration
- Dashboard Customization
- Role-Based Access Control
- Dataset Sharing

---

# 👩‍💻 Author

**Bhumika Singh**

B.Tech CSE, VIT Vellore

GitHub: https://github.com/bhumika0810

---

## ⭐ If you found this project interesting, consider giving it a star!
