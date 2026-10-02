from src.data_loader import Load_all_documents
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
if __name__=="__main__":

    #docs = Load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    ##print(store.query("What is Deep Learning",top_k=3))

    rag_serach=RAGSearch()
    query="What is Deep Learning"
    summary=rag_serach.search_and_summarize(query,top_k=3)
    print("Summary:",summary)

