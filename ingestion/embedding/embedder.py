from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class LocalHuggingFaceEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings wrapper for SentenceTransformers.
    This allows us to use sentence-transformers directly without adding extra heavy dependencies.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the model directly using sentence-transformers
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (texts)."""
        # encode returns numpy arrays; we convert to lists of floats
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """
    Returns a LangChain-compatible embedding model instance.
    Defaults to all-MiniLM-L6-v2, which is lightweight and fast for V1.
    """
    print(f"Loading embedding model: {model_name}...")
    return LocalHuggingFaceEmbeddings(model_name=model_name)
