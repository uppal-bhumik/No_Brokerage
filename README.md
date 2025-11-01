# AI Property Search Chatbot (NoBrokerage.com)

This is a full-stack AI chatbot built for the AI Engineer Intern assignment. The application allows users to search for properties using natural language queries and receive AI-generated, data-grounded summaries, along with a list of relevant property results.

The entire project was built and tested in under 3 days, focusing on a robust backend (AI, data) and a polished, functional frontend (UI).

---

## 🚀 Live Demo

https://no-brokerage.streamlit.app/

---

## 📹 Loom Demo

https://www.loom.com/share/17f255ce022c4f91aaff090a842a9905

---

## ✨ Core Features

* **Natural Language Querying:** Understands complex queries like "3 bhk in Mumbai under 2 crore" or "show me a ready to move studio apartment".
* **Intelligent Entity Extraction:** Uses `gpt-4o-mini` to parse user queries into structured JSON filters (`city`, `bhk`, `budget_max_lakhs`, `readiness`).
* **Data-Grounded Summaries (RAG):** Employs a RAG (Retrieval-Augmented Generation) pattern. The AI generates summaries *only* based on the data retrieved from the CSVs, preventing hallucinations.
* **Smart Query Handling:**
    * **"No Results"**: Provides an empathetic, helpful response.
    * **"Broad Query"**: (e.g., "what are the cheapest properties?") Understands the intent and provides a statistical summary (price range, count, etc.).
* **Robust Data Pipeline:**
    * Loads and merges 4 separate CSV files into a master database using Pandas.
    * Includes robust data cleaning for prices (e.g., "1.5 Cr", "90L", `9000000` -> all converted to lakhs) and BHKs (e.g., "Studio" -> `0`).
* **Polished UI:** A simple, user-friendly Streamlit interface with a chat history, loading spinners, and custom-styled, expandable property cards.

---

## 🛠️ Tech Stack

| Layer | Technology | Reason |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** | High-performance, async-first API framework. |
| **Frontend** | **Streamlit** | Allowed by prompt; enables rapid, clean UI development. |
| **AI / NLP** | **OpenAI (gpt-4o-mini)** | State-of-the-art model for fast, accurate JSON extraction and summary. |
| **Data Handling**| **Pandas** | The ideal tool for loading, merging, cleaning, and filtering the CSV data. |
| **API Client** | **Requests** | For connecting the Streamlit frontend to the FastAPI backend. |

---

## 🏃‍♂️ How to Run

This project consists of two separate applications (backend and frontend) that must be run in two separate terminals.

### 1. Prerequisites

* Python 3.8+
* An OpenAI API Key

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/uppal-bhumik/No_Brokerage.git
    cd https://github.com/uppal-bhumik/No_Brokerage
    ```

2.  **Create the environment file:**
    * In the root folder, create a new file named `.env`.
    * Copy the contents of `.env.example` into it and add your OpenAI API key:
    ```ini
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

3.  **Install dependencies for both apps:**
    ```bash
    # Install backend requirements
    pip install -r backend/requirements.txt
    
    # Install frontend requirements
    pip install -r frontend/requirements.txt
    ```

### 3. Launch the Application

**Terminal 1: Run the Backend**
```bash
python backend/main.py
```

Wait for the terminal to show Uvicorn running on http://0.0.0.0:8000

**Terminal 2: Run the Frontend**

```bash
streamlit run frontend/app.py
```
Your browser will automatically open to http://localhost:8501.

You can now use the chatbot.
