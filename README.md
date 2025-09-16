# 🎓 Student Assistant
An AI-powered academic assistant web app designed to help students manage **subjects, topics, and study materials**.  
It integrates a **chatbot interface**, **MongoDB persistence**, and **file-based context (CSV/PDF)** to enhance learning.  

---

## 🚀 Features

### 💬 AI-Powered Chatbot
- Built using **Google Generative AI (Gemini 1.5 Flash)** with LangChain.
- Understands natural language queries for managing subjects & topics.
- Integrates tools for:
  - Adding/Deleting subjects
  - Adding/Listing/Deleting topics
  - Finding which subject a topic belongs to
  - Bulk topic operations (all/multiple subjects)

### 📚 Subject & Topic Management
- Persistent storage in **MongoDB**.
- Supports **subject creation, deletion, and listing**.
- Allows **adding/removing topics** under specific or multiple subjects.

### 📂 File Upload Support
- Upload **CSV files** (automatically parsed into a Pandas DataFrame).
- Upload **PDF files** (text extracted via PyMuPDF).
- Data can be used as **context for chatbot queries**.

### 🗄️ Database Integration
- MongoDB stores all subjects and topics.
- Ensures **no duplicates** with `$addToSet`.
- Supports flexible queries for subject-topic relationships.

### 🧾 Planned DataFrame Agent (WIP)
- Uses `create_pandas_dataframe_agent` to query CSV data with natural language.
- Will allow Q&A over uploaded datasets (future enhancement).

---

## 🛠️ Tech Stack

| Category       | Technologies Used |
|----------------|------------------|
| Frontend       | Streamlit (Chat UI + Sidebar) |
| Backend        | LangChain + LangGraph |
| Database       | MongoDB (via PyMongo) |
| LLM            | Google Generative AI (Gemini 1.5 Flash) |
| File Parsing   | Pandas (CSV), PyMuPDF (PDF) |
| Environment    | Python (dotenv for config) |

---

## 🧩 Project Structure
student-assistant/
│── study_dep.py # Core Graph & Tool setup
│── app.py # Streamlit chatbot UI
│── requirements.txt # Dependencies
│── .env # API keys & Mongo URI
│── README.md # Documentation


---

## 🔌 API Keys & Configuration

### 🔐 Required Keys:
- **Google Generative AI API Key** (for LLM responses)  
- **MongoDB URI** (for subject/topic persistence)  

### 📁 Add `.env` file:
```bash
GOOGLE_API_KEY=your_gemini_key
MONGO_URI=your_mongodb_uri

