# 📦 Supply Chain News AI Research Tool (RAG Assistant)

An intelligent **Retrieval-Augmented Generation (RAG)** assistant designed to aggregate, process, search, and analyze real-time supply chain news.

The system combines **web scraping, semantic search, vector databases, transformer embeddings, and high-speed LLM inference** to help supply chain analysts quickly extract insights from unstructured market intelligence.

Built with:

- 🕷️ `Cloudscraper` + `BeautifulSoup4` for web data ingestion
- 🧩 `LangChain` for RAG orchestration
- 🤗 Hugging Face `all-MiniLM-L6-v2` for semantic embeddings
- 🔎 `FAISS` for vector similarity search
- ⚡ `Groq` + `Llama 3.3 70B` for fast LLM inference
- 🎨 `Streamlit` for the interactive research interface

---

## 🎯 Project Overview

Supply chain intelligence is often scattered across news articles, industry reports, and constantly changing web sources.

Traditional keyword search can make it difficult to quickly answer questions such as:

> "What are the latest disruptions affecting global semiconductor supply chains?"

> "What factors are currently driving shipping delays?"

> "What supply chain risks are being reported across major industries?"

This project addresses that problem by creating an AI-powered research assistant that can:

**Collect → Clean → Chunk → Embed → Retrieve → Generate → Cite**

Instead of relying solely on an LLM's internal knowledge, the assistant retrieves relevant information from the indexed supply chain news before generating its response.

---

# 🏗️ Architecture & Workflow

<img width="2391" height="8192" alt="Data Ingestion to RAG-2026-09-27-133238" src="https://github.com/user-attachments/assets/46281eb2-dedb-4ca1-b41d-d3aca00b2d48" />

The assistant follows a standard **5-stage Retrieval-Augmented Generation workflow**.

### 1. 🌐 Data Ingestion

**File:** `data_ingestion.py`

The ingestion layer collects supply chain news from relevant websites using:

- `cloudscraper`
- `BeautifulSoup4`
- HTTP requests
- HTML parsing

The goal is to transform unstructured web content into usable documents containing article text and source metadata.

```text
Web Sources
     ↓
Cloudscraper
     ↓
HTML Content
     ↓
BeautifulSoup4
     ↓
Clean Article Data
2. ✂️ Text Chunking & Preprocessing

File: 02_text_chunking.py

Long-form articles are divided into smaller, manageable text chunks using LangChain's:

RecursiveCharacterTextSplitter

Chunking is important because sending entire articles to an LLM can introduce unnecessary context and exceed model context limits.

Instead, the articles are broken into smaller pieces that can later be searched independently.

Long Article
     ↓
Text Cleaning
     ↓
Recursive Character Splitting
     ↓
Smaller Text Chunks

For example:

Original Article
────────────────────────────
Supply chains around the
world are experiencing...
[hundreds of additional words]
────────────────────────────

              ↓

Chunk 1
────────────────────────────
Supply chains around the
world are experiencing...
────────────────────────────

Chunk 2
────────────────────────────
Shipping costs have increased...
────────────────────────────

Chunk 3
────────────────────────────
Manufacturers are responding...
────────────────────────────

This makes the information easier to retrieve later.

3. 🧠 Vector Embeddings & Storage

File: 03_vector_store.py

Each text chunk is converted into a numerical representation called an embedding.

The project uses the Hugging Face model:

sentence-transformers/all-MiniLM-L6-v2

The embedding model converts text into vectors that represent the semantic meaning of the text.

These vectors are then stored in a local:

FAISS (Facebook AI Similarity Search)

index.

The process can be summarized as:

Text Chunk
     ↓
Hugging Face Embedding Model
     ↓
Numerical Vector
     ↓
FAISS Vector Index

For example, these two questions use different words but have similar meanings:

"What is causing shipping delays?"

"What factors are slowing global logistics?"

A semantic embedding model can recognize that these questions are conceptually related even though the exact words are different.

4. 🔎 Retrieval-Augmented Generation

File: 04_rag_chain.py

This is the core of the application.

When a user asks a question, the system:

Converts the question into an embedding.
Searches the FAISS vector index.
Retrieves the most relevant document chunks.
Combines the retrieved context with the user's question.
Sends the contextual prompt to the LLM.
Generates an analytical response.

The LLM is powered by:

Groq Cloud — llama-3.3-70b-versatile

The simplified process is:

User Question
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Document Chunks
      ↓
Context + User Question
      ↓
Groq LLM
      ↓
AI-Generated Response

This is the fundamental idea behind Retrieval-Augmented Generation.

Instead of asking the LLM to answer purely from its pretrained knowledge, the application retrieves relevant information from its external knowledge base and provides that information as context.

5. 🎨 Interactive UI

File: 05_app.py

The final application is built with Streamlit.

The interface allows users to submit natural-language questions and receive AI-generated research responses based on the indexed supply chain news.

The application is deployed through Streamlit Cloud.

User
 ↓
Streamlit Interface
 ↓
RAG Pipeline
 ↓
FAISS Retrieval
 ↓
Groq LLM
 ↓
Research Response
✨ Features
🕷️ Web Data Ingestion

Collects unstructured supply chain information from online sources using:

cloudscraper
BeautifulSoup4
Python-based HTTP requests
HTML parsing

The ingestion process transforms raw web pages into documents suitable for downstream processing.

🧠 Semantic Vector Search

Instead of relying only on keyword matching, the application uses transformer-based embeddings to search for information based on meaning and semantic similarity.

This means the system can potentially identify relevant content even when the wording of the question differs from the wording used in the source article.

🔎 FAISS Similarity Search

FAISS provides fast similarity search over the generated document embeddings.

This allows the application to retrieve relevant pieces of information before sending them to the LLM.

🤖 Retrieval-Augmented Generation

The application combines:

User Question + Retrieved Context + LLM

to generate a contextual response.

This approach is particularly useful for domain-specific research where the model needs access to information outside its pretrained knowledge.

⚡ High-Speed LLM Inference

The project uses Groq Cloud for LLM inference with:

llama-3.3-70b-versatile

The goal is to provide fast responses while maintaining the reasoning and language capabilities of a large language model.

🎨 Interactive Web Application

The Streamlit interface provides a simple way for users to interact with the RAG system without needing to run individual Python scripts manually.

🛠️ Tech Stack
Category	Technology
Programming Language	Python
Frontend / UI	Streamlit
RAG Orchestration	LangChain
LLM	Groq — llama-3.3-70b-versatile
Embeddings	Hugging Face — sentence-transformers/all-MiniLM-L6-v2
Vector Database	FAISS
Web Scraping	Cloudscraper
HTML Parsing	BeautifulSoup4
Deployment	Streamlit Cloud
Environment Management	Python venv + .env
🚀 Live Demo

[!NOTE]
The application is deployed and available for interactive testing.

<img width="1918" height="933" alt="Supply Chain RAG Assistant" src="https://github.com/user-attachments/assets/f01d4c5a-8a81-45af-888b-180652033f16" />
🔗 Live Application

Supply Chain RAG Assistant on Streamlit

📂 Project Structure
supply-chain-rag-assistant/
│
├── faiss_index/
│   └── ...                  # Local FAISS vector database
│
├── .env                     # API keys and environment variables
├── .gitignore               # Git exclusion rules
│
├── data_ingestion.py        # Web scraping & document ingestion
├── 02_text_chunking.py      # Text cleaning & document chunking
├── 03_vector_store.py       # Embedding generation & FAISS indexing
├── 04_rag_chain.py          # RAG retrieval & LLM pipeline
├── 05_app.py                # Streamlit application
│
├── test_connection.py       # API connectivity testing
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
⚙️ Local Setup & Installation
1. Prerequisites

Before running the project locally, make sure you have:

Python 3.10 or higher
Git
A Groq API key
Internet access for web data ingestion and API communication
2. Clone the Repository
git clone https://github.com/tinotendamarufetu/supply-chain-rag-assistant.git

cd supply-chain-rag-assistant
3. Create a Virtual Environment
Windows PowerShell
python -m venv venv

.\venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv venv

source venv/bin/activate
4. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt
5. Configure Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

[!WARNING]
Never commit API keys or other secrets to GitHub.

Make sure .env is included in .gitignore:

.env
venv/
__pycache__/
6. Run the Application

Start the Streamlit application:

streamlit run 05_app.py

Streamlit will provide a local URL where you can access the application in your browser.

🔬 Understanding the RAG Pipeline

The entire system can be summarized as:

                    SUPPLY CHAIN NEWS
                           │
                           ▼
                  ┌─────────────────┐
                  │  DATA INGESTION │
                  │                 │
                  │ Cloudscraper    │
                  │ BeautifulSoup   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ TEXT PROCESSING │
                  │                 │
                  │ Cleaning        │
                  │ Chunking        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   EMBEDDINGS    │
                  │                 │
                  │ Hugging Face    │
                  │ MiniLM-L6-v2    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ FAISS VECTOR    │
                  │ DATABASE        │
                  └────────┬────────┘
                           │
                           │
                    USER QUESTION
                           │
                           ▼
                  ┌─────────────────┐
                  │ QUERY EMBEDDING │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ SEMANTIC SEARCH │
                  │                 │
                  │ FAISS Retrieval │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ RELEVANT        │
                  │ CONTEXT         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     GROQ LLM    │
                  │                 │
                  │ Llama 3.3 70B   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ AI RESEARCH     │
                  │ RESPONSE        │
                  └─────────────────┘
🧩 What Happens When a User Asks a Question?

Imagine the user asks:

"What are the latest supply chain disruptions affecting the semiconductor industry?"

The application processes the question approximately like this:

Step 1 — Receive the Question

The Streamlit application receives the user's natural-language question.

"What are the latest supply chain disruptions
affecting the semiconductor industry?"
Step 2 — Create a Query Embedding

The question is converted into an embedding using:

sentence-transformers/all-MiniLM-L6-v2
Step 3 — Search FAISS

The query vector is compared against the vectors stored in the FAISS index.

The system identifies document chunks that are semantically similar to the question.

Step 4 — Retrieve Context

The most relevant chunks are returned to the RAG pipeline.

User Question
      +
Retrieved Article Chunks
      ↓
Contextual Prompt
Step 5 — Generate the Response

The retrieved context is provided to the Groq-powered LLM.

The model then generates a natural-language response based on the available context.

🧠 Why Use RAG?

Large Language Models are powerful, but they have an important limitation:

An LLM does not automatically have access to every new piece of information published on the internet.

For a rapidly changing domain such as supply chain intelligence, this creates a challenge.

A traditional LLM workflow might look like:

User Question
      ↓
LLM
      ↓
Answer

A RAG workflow looks different:

User Question
      ↓
Search External Knowledge
      ↓
Retrieve Relevant Information
      ↓
Provide Context to LLM
      ↓
Generate Response

This architecture allows an application to connect an LLM to an external knowledge source.

🔍 Keyword Search vs. Semantic Search

Traditional keyword search might look for an exact word such as:

"shipping delays"

Semantic search attempts to understand the underlying meaning.

For example:

Question 1:
"What is causing shipping delays?"

Question 2:
"What factors are slowing global logistics?"

Question 3:
"Why are goods taking longer to reach customers?"

These questions use different language but are conceptually related.

Embedding models allow the application to represent these sentences numerically and compare their semantic similarity.

📚 Example Research Questions

The assistant can be used to explore questions such as:

What are the latest supply chain disruptions affecting the semiconductor industry?

What factors are currently contributing to global shipping delays?

What supply chain risks are emerging in the automotive industry?

What recent events could impact global logistics costs?

Which industries are experiencing supply chain pressure?

What are the major supply chain trends reported in recent news?

What factors are affecting global manufacturing?

What disruptions are being reported in international trade?

How are companies responding to logistics challenges?

What recent events could affect supplier risk?
💼 Potential Business Applications

Although this project focuses on supply chain news, the underlying RAG architecture can be adapted to many enterprise use cases.

🚢 Supply Chain Intelligence

Potential applications include monitoring:

Logistics disruptions
Shipping delays
Supplier risks
Commodity markets
Manufacturing disruptions
Transportation issues
Global trade developments
💰 Financial Research

The same architecture could be adapted to search:

Market news
Earnings reports
Economic releases
Company filings
Financial research
Industry reports
🏢 Enterprise Knowledge Search

Organizations could adapt this architecture to search internal information such as:

Company documentation
Policies
Technical manuals
Customer support documentation
Research reports
Operational procedures
Knowledge bases
🔮 Future Improvements

There are several ways this project could be extended.

Data Engineering Improvements
 Automated scheduled news ingestion
 Additional supply chain news sources
 Automated article deduplication
 Data quality validation
 Metadata enrichment
 Historical news storage
 Automated source monitoring
Retrieval Improvements
 Hybrid keyword + vector search
 Metadata filtering
 Document re-ranking
 Improved chunking strategies
 Retrieval evaluation
 Semantic caching
 Persistent production vector database
AI Improvements
 Conversation memory
 Multi-document summarization
 News sentiment analysis
 Supply chain risk classification
 Automated trend detection
 Entity extraction
 Question decomposition
 RAG evaluation using RAGAS or similar frameworks
Production Improvements
 Cloud-based data pipeline
 Scheduled workflows
 API endpoint
 Authentication
 Logging and monitoring
 Error tracking
 Automated testing
 CI/CD pipeline
🧪 Possible Evaluation Framework

A future version of the project could evaluate the RAG pipeline using metrics such as:

Retrieval Quality

Measure whether the system retrieves the correct documents for a given question.

Context Relevance

Measure whether the retrieved context is actually relevant to the question.

Answer Faithfulness

Measure whether the generated answer is supported by the retrieved context.

Answer Relevance

Measure whether the final response actually addresses the user's question.

This would allow the project to move beyond simply demonstrating that a RAG pipeline works and toward measuring how well the RAG system works.

🎓 Key Learning Outcomes

This project provided hands-on experience with several important concepts in modern AI application development.

Retrieval-Augmented Generation

Understanding how external knowledge can be connected to an LLM to create domain-specific AI applications.

Vector Embeddings

Learning how natural-language text can be transformed into numerical representations that capture semantic meaning.

Vector Search

Working with FAISS to efficiently retrieve semantically similar document chunks.

Prompt Engineering

Designing prompts that combine user questions with retrieved contextual information.

LLM Integration

Connecting a Python application to an external LLM inference provider through Groq.

Data Engineering

Building a pipeline that transforms messy web data into structured information suitable for downstream AI applications.

Web Scraping

Collecting and parsing information from publicly accessible web sources.

AI Application Development

Combining multiple AI and data technologies into a complete end-to-end application.

Application Deployment

Moving a local Python prototype into an interactive Streamlit application.

🏗️ End-to-End Architecture

At a high level, the project demonstrates how multiple technologies can work together as a complete AI system:

┌───────────────────────┐
│   Supply Chain News   │
│      Web Sources      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Data Ingestion     │
│  Cloudscraper + BS4   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Text Processing &    │
│       Chunking        │
│      LangChain        │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      Embeddings       │
│     Hugging Face      │
│   all-MiniLM-L6-v2    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     FAISS Index       │
│   Vector Retrieval    │
└───────────┬───────────┘
            │
            │
      ┌─────▼─────┐
      │ User Query │
      └─────┬─────┘
            │
            ▼
┌───────────────────────┐
│   Semantic Retrieval  │
│    Relevant Context   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│       Groq LLM        │
│   Llama 3.3 70B       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Streamlit Research  │
│       Assistant       │
└───────────────────────┘
🔐 Security

API credentials should never be committed to the repository.

Use environment variables instead:

GROQ_API_KEY=your_api_key_here

Your .gitignore should include:

.env
venv/
__pycache__/
*.pyc

If an API key is accidentally exposed, revoke it immediately and generate a new one.

⚠️ Important Notes

This project is intended as a research and educational AI application.

The quality of generated responses depends on:

The quality of the scraped sources
The completeness of the indexed documents
The chunking strategy
The embedding model
The retrieval configuration
The LLM
The quality of the user's question

Generated responses should therefore be treated as AI-assisted research outputs, rather than automatically verified facts.

Users should consult the underlying sources when making important business or operational decisions.

📌 Project Highlights

This project demonstrates an end-to-end AI pipeline rather than an isolated machine learning model.

The complete workflow covers:

Web Data
   ↓
Data Ingestion
   ↓
Text Processing
   ↓
Semantic Embeddings
   ↓
Vector Database
   ↓
Information Retrieval
   ↓
LLM Generation
   ↓
Interactive Application
   ↓
Cloud Deployment

This makes the project a practical example of how Data Engineering + Information Retrieval + Generative AI + Application Development can be combined into one system.

🚀 Future Vision

The long-term direction of the project is to evolve the prototype into a more comprehensive Supply Chain Intelligence Platform.

Potential capabilities could include:

                    SUPPLY CHAIN INTELLIGENCE
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     News Search         Risk Detection      Trend Analysis
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                     AI Research Assistant
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
      Summaries          Risk Alerts        Insights

Possible future features include:

Real-time monitoring
Supply chain risk alerts
Industry-specific dashboards
Automated daily intelligence reports
News clustering
Event detection
Trend analysis
Supplier risk monitoring
AI-generated executive summaries
📄 License

Distributed under the MIT License.

See the LICENSE file for more information.

👨‍💻 Author
Tinotenda Muchenje

Data & AI professional interested in building practical technology solutions across:

📊 Data Analytics
📈 Business Intelligence
🤖 Generative AI
🔎 Retrieval-Augmented Generation
🧠 Machine Learning
🏗️ Data Engineering
☁️ Cloud Architecture
⚙️ Automation
⭐ Support the Project

If you find this project useful or interesting:

⭐ Star the repository
🍴 Fork the project
💬 Share feedback
🧑‍💻 Experiment with the code
🚀 Build your own RAG application

Built with Python, LangChain, FAISS, Hugging Face, Groq, and Streamlit.

Turning unstructured supply chain information into searchable, AI-assisted intelligence.
