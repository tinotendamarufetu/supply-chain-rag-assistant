import os
import streamlit as st
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (.env)
load_dotenv()

# 1. Page Config
st.set_page_config(
    page_title="Supply Chain News Research Tool",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain News AI Research Assistant")
st.caption("Grounded RAG System powering market intelligence & supply chain risk analysis")


# 2. Helper Function: Load & Scrape Article URLs using cloudscraper
def scrape_url(url: str) -> str:
    """Scrapes raw web page text using cloudscraper to bypass anti-bot protection."""
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract main text paragraphs
        paragraphs = soup.find_all("p")
        text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        return text_content
    else:
        raise ValueError(f"Failed to fetch content from {url} (Status: {response.status_code})")


# 3. Sidebar UI: Custom URL Input & Ingestion Pipeline
st.sidebar.title("News Article URLs")

url1 = st.sidebar.text_input("URL 1", placeholder="https://www.supplychain.news/...")
url2 = st.sidebar.text_input("URL 2", placeholder="https://www.supplychain.news/...")
url3 = st.sidebar.text_input("URL 3", placeholder="https://www.supplychain.news/...")

process_url_clicked = st.sidebar.button("Process URLs")

# Shared Embedding Model Setup
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Handle URL Ingestion, Chunking, and FAISS Vectorstore Creation
if process_url_clicked:
    urls = [u.strip() for u in [url1, url2, url3] if u.strip()]
    
    if not urls:
        st.sidebar.error("⚠️ Please insert at least one valid URL.")
    else:
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("⏳ Scraping web content...")
        
        documents = []
        for url in urls:
            try:
                content = scrape_url(url)
                if content:
                    doc = Document(page_content=content, metadata={"source": url})
                    documents.append(doc)
            except Exception as e:
                st.sidebar.warning(f"Failed to scrape {url}: {e}")
        
        if documents:
            status_placeholder.info("✂️ Chunking text data...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)

            status_placeholder.info("⚡ Building FAISS Vector Index...")
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local("faiss_index")
            
            status_placeholder.success("✅ FAISS Vector Index Created Successfully!")
        else:
            status_placeholder.error("❌ Could not extract text from the provided URLs.")


# 4. Load Active Vector Database
@st.cache_resource(show_spinner=False)
def load_vector_db():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None


# 5. Chat & Query System
groq_api_key = os.getenv("GROQ_API_KEY")
vector_db = load_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Enter article URLs in the sidebar, click 'Process URLs', and ask any question about them here."
        }
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 View Cited Sources"):
                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")

# User Query Processing
if user_query := st.chat_input("Ask a question about the processed supply chain news..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        if not os.path.exists("faiss_index"):
            response_text = "⚠️ No processed data found! Please add URLs in the sidebar and click 'Process URLs' first."
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        elif not groq_api_key:
            st.error("❌ GROQ_API_KEY not found in .env file.")
        else:
            with st.spinner("Analyzing articles & generating response..."):
                try:
                    # Reload local vector index
                    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    retriever = db.as_retriever(search_kwargs={"k": 3})

                    # Initialize LLM
                    llm = ChatGroq(
                        model="openai/gpt-oss-120b",
                        api_key=groq_api_key,
                        temperature=0.2,
                    )

                    # Context Format Helper
                    def format_docs(docs):
                        return "\n\n".join(
                            [f"Source ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for doc in docs]
                        )

                    system_prompt = (
                        "You are an expert Supply Chain Market Research Assistant.\n"
                        "Use ONLY the following retrieved context snippets to answer the user's question.\n"
                        "If the answer is not contained within the provided context, state clearly: "
                        "'I cannot find information about this in the provided articles.'\n\n"
                        "Context:\n{context}\n\n"
                        "Question: {question}"
                    )
                    prompt = ChatPromptTemplate.from_template(system_prompt)

                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Retrieve context & run chain
                    retrieved_docs = retriever.invoke(user_query)
                    sources = list(set([doc.metadata.get("source") for doc in retrieved_docs if doc.metadata.get("source")]))

                    answer = rag_chain.invoke(user_query)

                    # Output answer
                    st.write(answer)

                    # Output collapsible sources
                    if sources:
                        with st.expander("📌 View Cited Sources"):
                            for source in sources:
                                st.markdown(f"- [{source}]({source})")

                    # Append to session state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
