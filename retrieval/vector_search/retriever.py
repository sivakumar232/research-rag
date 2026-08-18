import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from typing import List, Dict, Any

# Ensure project root is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.embedding.embedder import get_embedding_model
    
# Load environment variables for Chroma Cloud
load_dotenv()

def get_chroma_collection(collection_name: str = "upsc_gs_v1"):
    """
    Connects to ChromaDB Cloud and retrieves our existing collection.
    """         
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")
    
    # Initialize the Cloud Client
    chroma_client = chromadb.CloudClient(
        tenant=chroma_tenant,
        database=chroma_database,
        api_key=chroma_api_key
    )
    
    # Get the collection we created during Ingestion
    return chroma_client.get_collection(name=collection_name)


def retrieve_documents(query: str, top_k: int = 5, collection_name: str = "upsc_gs_v1") -> List[Dict[str, Any]]:
    """
    Searches the Vector Database for the chunks most relevant to the user's query.
    """
    # 1. Connect to the database
    collection = get_chroma_collection(collection_name)
    
    # 2. Load the EXACT SAME embedding model we used during ingestion
    embedder = get_embedding_model()
    
    # 3. Convert the user's text question into a math Vector
    print(f"Embedding query: '{query}'...")
    query_embedding = embedder.embed_query(query)
    
    # 4. Search the database!
    print(f"Searching ChromaDB collection '{collection_name}' for top {top_k} matches...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 5. Format the results nicely to return
    formatted_results = []
    
    # Chroma returns lists of lists (since you can query multiple questions at once)
    # We only asked 1 question, so we grab the first item [0] from the results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] # The 'math score' of how close it is
    
    for i in range(len(documents)):
        formatted_results.append({
            "text": documents[i],
            "metadata": metadatas[i],
            "score": distances[i]
        })
        
    return formatted_results


if __name__ == "__main__":
    # A quick test to see if it works!
    test_question = "How many moons does earth have and how far it is"
    print("\n--- Testing Retriever ---")
    matches = retrieve_documents(test_question, top_k=10)
    
    print("\n--- Top 10 Results ---")
    for idx, match in enumerate(matches):
        print(f"\nResult #{idx + 1} (Score: {match['score']:.4f})")
        print(f"Source: {match['metadata'].get('source', 'Unknown')}")
        print(f"Doc Name: {match['metadata'].get('doc_name', 'Unknown')}")
        print(f"Batch: {match['metadata'].get('batch', 'Unknown')}")
        print(f"Preview: {match['text'][:200]}...")
