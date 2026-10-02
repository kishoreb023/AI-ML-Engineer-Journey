import os

from src.vectorstore import FaissVectorStore
from langchain_ollama import ChatOllama


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen2.5:7b"
    ):

        # Create FAISS vector store
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Check whether FAISS already exists
        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        # If FAISS doesn't exist, build it
        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from src.data_loader import Load_all_documents

            docs = Load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        # Otherwise load existing FAISS
        else:

            self.vectorstore.load()

        # Initialize local Ollama LLM
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0
        )

        print(
            f"[INFO] Ollama LLM initialized: {llm_model}"
        )

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Search FAISS for relevant chunks
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # Extract text from retrieved chunks
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        # Combine chunks into context
        context = "\n\n".join(texts)

        # If nothing was found
        if not context:
            return "No relevant documents found."

        # Create prompt for local LLM
        prompt = f"""
Answer the question using only the provided context.

Question:
{query}

Context:
{context}

Instructions:
- Give a clear and simple answer.
- Use only information from the context.
- If the answer is not present in the context, say:
  "The answer is not available in the provided documents."

Answer:
"""

        # Send prompt to Ollama
        response = self.llm.invoke(prompt)

        return response.content


# Example usage
if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is attention mechanism?"

    answer = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("\nAnswer:")
    print(answer)