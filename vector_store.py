from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import persist_directory

def initialize_vector_store(documents):
    embedding = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    chroma = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    chroma.add_documents(documents)
    
    return chroma
