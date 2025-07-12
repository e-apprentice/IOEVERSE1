# IOEverse
Learning platform targeted towards IOE students


**IOEverse** is a cutting-edge educational platform built specifically for students of the **Institute of Engineering (IOE), Nepal**. It offers personalized learning experiences using **Generative AI** and **gamified quizzes**, helping students not only understand their syllabus content better but also engage actively in their studies.

## Features

### RAG-based Content Summarization & Quiz Generation
- Uses a **Retrieval-Augmented Generation (RAG)** approach to create quizzes directly from structured syllabus notes.
- When a user selects a topic or asks a question:
  - **Contextual multiple-choice questions (MCQs)** are generated from the relevant content.
  - After the user selects an answer, an **AI-generated explanation** is provided based on the same content.



### AI Chat Assistant (Gemini-powered)
- Get instant academic support through a conversational chatbot.
- Powered by **Gemini Pro**, capable of answering complex queries across engineering subjects.
- Provides definitions, concept explanations, and problem-solving guidance.  

### Interactive, Gamified Quizzes
- After reviewing summarized notes, students can test their knowledge through dynamically generated quizzes.
- Questions range from Easy to Advanced, with explanations for each answer.

### Student-Friendly and Secure
- No personal data collection beyond basic login (OAuth or email).
- Built for low-latency environments with mobile-first accessibility.

---

## Tech Stack

| Component         | Technology |
|------------------|--------------------------------------|
| Backend           | Python (Flask)                      |
| AI Chatbot        | Gemini via LangChain & Google GenAI |
| RAG Pipeline      | FAISS  + LangChain                  |
| Frontend          | HTML+ CSS + TailwindCSS              |
| Quiz Generator    | RAG-based MCQ generation engine with explanations |
| Database          | PostgreSQL           |

---

##  How to Run Locally

### Prerequisites
- Python 3.9+
- Node.js 18+
- Google API Key & Project for Gemini
- FAISS or ChromaDB setup

### Backend
```bash
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload

###  Frontend Setup

To run the frontend:

1. You can double-click it or open it with a live server extension (e.g., in VS Code).
2. Make sure the backend is running to enable features like quizzes and the Gemini chatbot.

```bash
cd client
# Open index.html in your browser or editor

Project Structure:
FLASK2/
├── static/
│   └── image/
│       └── pro.jpg
├── templates/
│   ├── engineering.html
│   ├── index.html
│   ├── physics.html
│   ├── quiz.html
│   ├── semester.html
│   ├── signup.html
│   ├── subject.html
│   └── summary.html
├── .env
├── app.py
└── requirements.txt
