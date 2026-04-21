# VECTOR-LESS RAG 

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
import re

os.environ["GROQ_API_KEY"] = "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"

def load_pdf(pdf_path):
    print(f"\n Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages")
    return documents

# Split into chunks
def split_documents(documents):
    print("\n Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks")
    return chunks

# Keyword retrieval
def keyword_search(chunks, query, k=3):
    """
    Simple vector-less retrieval:
    ranks chunks based on keyword overlap
    """

    query_words = set(re.findall(r"\w+", query.lower()))
    scored_chunks = []

    for chunk in chunks:
        text = chunk.page_content.lower()
        words = set(re.findall(r"\w+", text))

        score = len(query_words & words)  

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for _, chunk in scored_chunks[:k]]

    return top_chunks

def setup_llm():
    print("\n Setting up LLM...")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",   
        temperature=0.2
    )
    print(" LLM ready")
    return llm

def ask_question(llm, chunks, question):
    print(f"\n Question: {question}")
    print("-" * 50)

    top_chunks = keyword_search(chunks, question, k=3)

    if not top_chunks:
        print(" No relevant context found.")
        return

    context = "\n\n".join([c.page_content for c in top_chunks])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print(f"\n Answer:\n{response.content}")

    print("\n Source Chunks Used:")
    for i, c in enumerate(top_chunks):
        print(f"\n [{i+1}] Page {c.metadata.get('page', 'N/A')}")
        print(c.page_content[:200] + "...")

def main():

    PDF_PATH = r"C:\Users\pssra\OneDrive\Desktop\research_paper.pdf"

    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)

    llm = setup_llm()

    print("\n" + "="*50)
    print(" VECTOR-LESS RAG READY")
    print(" Type 'exit' to quit")
    print("="*50)

    while True:
        question = input("\n You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print(" Goodbye!")
            break
        if question == "":
            continue

        ask_question(llm, chunks, question)

if __name__ == "__main__":
    main()