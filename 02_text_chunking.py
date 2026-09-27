import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the scraping function we created in Step 1
from data_ingestion import fetch_clean_article

load_dotenv()

def run_text_chunking():
    print("--- ✂️ Starting Supply Chain Text Chunking ---")

    supply_chain_urls = [
        "https://www.supplychaindive.com/news/shippers-are-exploring-port-to-inland-transit-to-de-risk-supply-chains/831261/",
        "https://www.supplychaindive.com/news/companies-channel-ieepa-tariff-refunds-into-their-coffers-atlanta-fed-says/831169/"
    ]

    # 1. Fetch raw documents using our Step 1 function
    print(f"\n[1/3] Loading raw documents from {len(supply_chain_urls)} URLs...")
    raw_documents = []
    for url in supply_chain_urls:
        try:
            doc = fetch_clean_article(url)
            raw_documents.append(doc)
        except Exception as e:
            print(f"  ❌ Failed to fetch {url}: {e}")

    print(f"✅ Loaded {len(raw_documents)} raw document(s).")

    # 2. Configure RecursiveCharacterTextSplitter
    # chunk_size: Max length of each chunk (characters/tokens)
    # chunk_overlap: Overlapping text between adjacent chunks to maintain context boundaries
    chunk_size = 1000
    chunk_overlap = 200

    print(f"\n[2/3] Splitting documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    # 3. Perform the split
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} text chunks from {len(raw_documents)} articles!")

    # 4. Learning Checkpoint: Inspect individual chunks
    print("\n--- 🔍 LEARNING CHECKPOINT: Split Chunk Inspection ---")
    for i in [0, 1, 2]: # Inspect first 3 chunks
        if i < len(chunks):
            chunk = chunks[i]
            print(f"\n🧩 Chunk #{i+1}:")
            print(f"  • Source URL: {chunk.metadata.get('source')}")
            print(f"  • Character Length: {len(chunk.page_content)}")
            print(f"  • Text Snippet:\n    \"{chunk.page_content[:250]}...\"")
            print("  " + "-" * 60)

if __name__ == "__main__":
    run_text_chunking()