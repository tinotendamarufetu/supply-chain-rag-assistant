import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")


def format_docs(docs):
    return "\n\n".join(
        [
            f"Source ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
            for doc in docs
        ]
    )


def run_rag_chain():
    print("--- 🤖 Starting Modern Supply Chain RAG Chain ---")

    # 2. Load the local FAISS Index
    print("\n[1/3] Loading saved FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize Groq LLM with active model string
    print("\n[2/3] Connecting to Groq API...")
    # 3. Initialize Groq LLM with an active production model
    print("\n[2/3] Connecting to Groq API...")
    llm = ChatGroq(
        model="openai/gpt-oss-120b",  # Active Groq model ID
        api_key=groq_api_key,
        temperature=0.2,
    )

    # 4. Construct Prompt Template
    system_prompt = (
        "You are an expert Supply Chain Market Research Assistant.\n"
        "Use ONLY the following retrieved context snippets to answer the user's question.\n"
        "If the answer is not contained within the provided context, state clearly: "
        "'I cannot find information about this in the provided articles.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # 5. Build RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute Query
    print("\n[3/3] Executing RAG Query...")
    query = "How are shippers using port-to-inland transit to de-risk supply chains?"
    print(f"  • User Query: \"{query}\"\n")

    try:
        retrieved_docs = retriever.invoke(query)
        answer = rag_chain.invoke(query)

        print("--- 💡 LEARNING CHECKPOINT: Grounded RAG Output ---")
        print(f"\n🤖 Answer:\n{answer}\n")

        print("📌 Sources Cited:")
        sources = set([doc.metadata.get("source") for doc in retrieved_docs])
        for src in sources:
            print(f"  • {src}")

    except Exception as e:
        print(f"❌ Execution Error: {e}")


if __name__ == "__main__":
    run_rag_chain()