import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import the scraping function from Step 1
from data_ingestion import fetch_clean_article
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def run_vector_store():
    print("--- 🧠 Starting Vector Embedding & FAISS Database Creation ---")

    supply_chain_urls = [
        "https://www.supplychaindive.com/news/shippers-are-exploring-port-to-inland-transit-to-de-risk-supply-chains/831261/",
        "https://www.supplychaindive.com/news/companies-channel-ieepa-tariff-refunds-into-their-coffers-atlanta-fed-says/831169/"
    ]

    # 1. Fetch & Chunk Documents (Steps 1 & 2 Combined)
    print(f"\n[1/4] Scraping and chunking {len(supply_chain_urls)} URLs...")
    raw_documents = [fetch_clean_article(url) for url in supply_chain_urls]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    # 2. Initialize Free Local Embedding Model
    print("\n[2/4] Initializing HuggingFace Embeddings ('all-MiniLM-L6-v2')...")
    # This downloads a small model (~90MB) on first run and executes 100% locally on CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Create FAISS Vector Store & Index Chunks
    print("\n[3/4] Generating embeddings and building local FAISS Vector Index...")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Save the FAISS index locally to disk so we don't have to re-embed every time
    db_path = "faiss_index"
    vector_db.save_local(db_path)
    print(f"✅ FAISS index successfully saved locally at ./{db_path}")

    # 4. LEARNING CHECKPOINT: Run Similarity Search Query
    print("\n--- 🔍 LEARNING CHECKPOINT: Semantic Similarity Search ---")
    query = "What are the benefits of inland port routes according to Brian Harold?"
    print(f"  • User Query: \"{query}\"")
    
    # Retrieve top 2 most semantically relevant chunks (k=2)
    results = vector_db.similarity_search(query, k=2)

    print(f"\n  🎯 Top {len(results)} Relevant Chunks Retrieved from FAISS:")
    for i, doc in enumerate(results, start=1):
        print(f"\n  Result #{i}:")
        print(f"    • Source: {doc.metadata.get('source')}")
        print(f"    • Content Snippet:\n      \"{doc.page_content[:300]}...\"")
        print("    " + "-" * 50)

if __name__ == "__main__":
    run_vector_store()