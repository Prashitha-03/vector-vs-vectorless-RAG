# vector-vs-vectorless RAG

This project compares two approaches of Retrieval-Augmented Generation (RAG): **Vector-based RAG** and **Vector-less RAG**.

   - **Vector RAG** uses embeddings and a vector database (ChromaDB) to perform semantic search and retrieve context.
   - **Vector-less RAG** relies on keyword matching to find relevant text without using embeddings.

The repository includes implementations of both approaches along with a simple dataset, allowing a clear comparison of their working, complexity, and performance.

**Key takeaway:** Vector RAG provides more accurate and context-aware results, while Vector-less RAG is simpler and easier to implement.
