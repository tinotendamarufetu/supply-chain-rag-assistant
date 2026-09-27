# 📦 Supply Chain News AI Research Tool (RAG Assistant)

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to aggregate, process, and query real-time supply chain news. By combining Web Scraping (`cloudscraper`), Vector Search (`FAISS`), Hugging Face Embeddings (`all-MiniLM-L6-v2`), and Groq's ultra-fast LLM inference (`llama-3.3-70b-versatile`), this tool enables supply chain analysts to query complex market intelligence with pinpoint accuracy.

---

## 🛠️ Architecture & Workflow

<img width="2391" height="8192" alt="Data Ingestion to RAG-2026-09-27-133238" src="https://github.com/user-attachments/assets/46281eb2-dedb-4ca1-b41d-d3aca00b2d48" />

The assistant follows a standard 5-stage RAG workflow:

1. **Data Ingestion (`data_ingestion.py`)**: Uses `cloudscraper` and `BeautifulSoup4` to bypass anti-bot protections and scrape key supply chain news websites.
2. **Text Chunking (`02_text_chunking.py`)**: Splits long-form articles into manageable semantic chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Vector Embedding & Storage (`03_vector_store.py`)**: Embeds text chunks using Hugging Face's `all-MiniLM-L6-v2` model and stores them in a local `FAISS` vector index.
4. **RAG Retrieval & Generation (`04_rag_chain.py`)**: Retrieves contextually relevant chunks and routes them alongside user queries to **Groq LLM** (`llama-3.3-70b-versatile`).
5. **Interactive UI (`05_app.py`)**: Deployed on **Streamlit Cloud** for seamless analyst interaction.

---

## ✨ Features

- **Anti-Bot Scraping**: Reliably fetches unstructured web data bypassing Cloudflare and standard bot protection.
- **Semantic Vector Search**: Sub-second context retrieval leveraging FAISS indexing and Hugging Face Embeddings.
- **Ultra-Fast LLM Inference**: Powered by Groq Cloud for instant, structured analytical responses.
- **Interactive Web App**: Simple Streamlit UI with clear query inputs and cited sources.

---

## 🚀 Live Demo

> [!NOTE]  
> <img width="1918" height="933" alt="image" src="https://github.com/user-attachments/assets/f01d4c5a-8a81-45af-888b-180652033f16" />

> *Upload a screenshot or GIF of the active Streamlit app interface here showing a query and response.*

- **Live Application**: [Supply Chain RAG Assistant on Streamlit](https://supply-chain-rag-assistant.streamlit.app)

---

## 📂 Project Structure

```text
├── faiss_index/          # Local FAISS vector database store
├── .env                  # Environment variables (API Keys - gitignored)
├── .gitignore            # Git exclusion rules
├── 02_text_chunking.py   # Document chunking & pre-processing logic
├── 03_vector_store.py    # Vector database initialization & embedding logic
├── 04_rag_chain.py       # LangChain RAG pipeline configuration
├── 05_app.py             # Main Streamlit web application entry point
├── data_ingestion.py     # Web scraping module (cloudscraper + BeautifulSoup)
├── requirements.txt      # Production dependencies
└── test_connection.py    # API connectivity verification script```

💻 Tech Stack
Frontend: Streamlit

Orchestration: LangChain

LLM Engine: Groq Cloud (llama-3.3-70b-versatile)

Embeddings: Hugging Face (sentence-transformers/all-MiniLM-L6-v2)

Vector Database: FAISS

Scraping: cloudscraper, beautifulsoup4

⚙️ Local Setup & Installation
1. Prerequisites
Python 3.10 or higher installed.

A free Groq API Key.

2. Clone the Repository
Bash
git clone [https://github.com/tinotendamarufetu/supply-chain-rag-assistant.git](https://github.com/tinotendamarufetu/supply-chain-rag-assistant.git)
cd supply-chain-rag-assistant
3. Set Up Virtual Environment
PowerShell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
4. Install Dependencies
Bash
pip install -r requirements.txt
5. Configure Environment Variables
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
6. Run the Application
Bash
streamlit run 05_app.py
📄 License
Distributed under the MIT License. See LICENSE for details.

