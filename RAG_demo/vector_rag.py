'''# ==========================================
# VECTOR RAG DEMO (WITH OLLAMA - LOCAL LLM)
# ==========================================

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ==============================
# STEP 1: SAMPLE DOCUMENTS
# ==============================

documents = [
    "Diabetes is a chronic disease that affects how the body processes blood sugar.",
    "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision.",
    "Yoga helps reduce stress and improves mental health.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks to solve complex problems."
]

# ==============================
# STEP 2: CHUNKING
# ==============================

chunks = documents  # already small

# ==============================
# STEP 3: LOAD EMBEDDING MODEL
# ==============================

print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# ==============================
# STEP 4: CREATE EMBEDDINGS
# ==============================

print("Generating embeddings...")
embeddings = embed_model.encode(chunks)

# ==============================
# STEP 5: STORE IN FAISS
# ==============================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index created with", index.ntotal, "documents")

# ==============================
# STEP 6: RETRIEVAL FUNCTION
# ==============================

def retrieve_docs(query, k=2):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    results = [chunks[i] for i in indices[0]]
    return results

# ==============================
# STEP 7: LLM USING OLLAMA
# ==============================

def generate_answer(context_docs, query):
    context = "\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()
        return result.get("response", "No response from model.")

    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# ==============================
# STEP 8: MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("\n=== VECTOR RAG DEMO (Ollama - Interactive Mode) ===")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Enter your question: ")

        # Exit condition
        if query.lower() == "exit":
            print("Exiting RAG demo...")
            break

        print("\nRetrieving relevant documents...")
        retrieved_docs = retrieve_docs(query)

        print("\nRetrieved Context:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{i}. {doc}")

        print("\nGenerating answer using LLM...")
        answer = generate_answer(retrieved_docs, query)

        print("\nFinal Answer:")
        print(answer)
        print("\n" + "-"*50 + "\n")'''


#Vector RAG

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os

os.environ["GROQ_API_KEY"] = "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"   

def load_pdf(pdf_path):
    print(f"\n Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages from PDF")
    return documents


# Split into chunks

def split_documents(documents):
    print("\n  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      
        chunk_overlap=50     
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks")
    return chunks


# Creating embeddings and storing in ChromaDB
def create_vector_store(chunks):
    print("\n Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"   
    )

    print(" Embeddings created and stored in ChromaDB")
    return vector_store

def load_vector_store():
    print("\n Loading existing ChromaDB vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    print(" Vector store loaded from disk")
    return vector_store


# LLM Setup
def setup_llm():
    print("\n Setting up LLM (Groq + LLaMA3)...")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2     
    )
    print(" LLM ready")
    return llm

def build_rag_chain(vector_store, llm):
    print("\n Building RAG chain...")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}    
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",              
        retriever=retriever,
        return_source_documents=True     
    )

    print(" RAG chain ready")
    return rag_chain

def ask_question(rag_chain, question):
    print(f"\n Question: {question}")
    print("-" * 50)

    result = rag_chain.invoke({"query": question})

    print(f" Answer:\n{result['result']}")
    print("\n Source Chunks Used:")

    for i, doc in enumerate(result['source_documents']):
        print(f"\n  [{i+1}] Page {doc.metadata.get('page', 'N/A')}:")
        print(f"       {doc.page_content[:200]}...")   

def main():

    PDF_PATH = r"C:\Users\pssra\OneDrive\Desktop\research_paper.pdf"

    if os.path.exists("./chroma_db"):
        print(" ChromaDB found — skipping embedding, loading from disk")
        vector_store = load_vector_store()
    else:
        print(" No ChromaDB found — building from PDF...")
        documents    = load_pdf(PDF_PATH)
        chunks       = split_documents(documents)
        vector_store = create_vector_store(chunks)

    llm = setup_llm()

    rag_chain = build_rag_chain(vector_store, llm)

    print("\n" + "="*50)
    print(" VECTOR RAG IS READY — Ask your questions!")
    print("   Type 'exit' to quit")
    print("="*50)

    while True:
        question = input("\n You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print(" Goodbye!")
            break
        if question == "":
            continue
        ask_question(rag_chain, question)

if __name__ == "__main__":
    main()