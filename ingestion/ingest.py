"""
Ingestion Pipeline (PDF Loading, Chunking, Embedding, & Cloud Storage)

This script manages loading PDFs from kb_pdf subfolders, converting them to Markdown, 
splitting them recursively, and saving them to ChromaDB Cloud.
"""

import os
import sys
import argparse
from pathlib import Path
import chromadb
from dotenv import load_dotenv

# Ensure project root is in sys.path for direct execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.loaders.pdf_loader import load_pdfs_from_directory
from ingestion.chunking.fixed_size import chunk_documents
from ingestion.embedding.embedder import get_embedding_model


def run_ingestion_pipeline(
    batch: str = "all",
    collection_name: str = "upsc_gs_v1",
    reset_collection: bool = False,
    limit: int | None = None
):
    # Load environment variables
    load_dotenv()
    
    # 1. Determine which directory to scan
    base_dir = "kb_pdf"
    if batch == "all" or batch == "root":
        raw_dir = base_dir
        recursive = (batch == "all")
    else:
        raw_dir = str(Path(base_dir) / batch)
        recursive = False
        if not Path(raw_dir).exists():
            print(f"Error: Directory '{raw_dir}' does not exist.")
            return

    # 2. Load and parse PDFs into Documents
    print(f"\n--- Loading and Parsing PDFs from: {raw_dir} ---")
    documents = load_pdfs_from_directory(
        raw_dir=raw_dir, 
        processed_dir="data/kb_processed", 
        recursive=recursive,
        limit=limit
    )
    if not documents:
        print("No documents were loaded. Exiting.")
        return

    # 3. Split documents recursively into chunks
    print(f"\n--- Chunking Documents (500 chars, 50 overlap) ---")
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50, use_recursive=True)
    print(f"Created {len(chunks)} chunks from {len(documents)} document(s).")

    # 4. Connect to ChromaDB Cloud
    print(f"\n--- Connecting to ChromaDB Cloud ---")
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")
    
    if not all([chroma_api_key, chroma_tenant, chroma_database]):
        print("Error: Missing ChromaDB Cloud credentials in your .env file.")
        return

    chroma_client = chromadb.CloudClient(
        tenant=chroma_tenant,
        database=chroma_database,
        api_key=chroma_api_key
    )

    # Reset collection if requested
    if reset_collection:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}' for a fresh start.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(collection_name)

    # 5. Embed and Upsert Chunks
    print(f"\n--- Embedding and Storing Chunks ---")
    embedder = get_embedding_model()
    
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]  # Deterministic IDs to avoid duplicates

    # Compute embeddings
    print("Generating embeddings (please wait)...")
    embeddings = embedder.embed_documents(texts)

    # Upload in batches of 200
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        end = i + batch_size
        print(f"Uploading chunks {i} to {min(end, len(chunks))}...")
        collection.upsert(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],  # type: ignore
            metadatas=metadatas[i:end]     # type: ignore
        )

    print(f"\nSuccess! Total collection count for '{collection_name}': {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean & Simple Ingestion CLI")
    parser.add_argument("-b", "--batch", default="all", help="Subfolder under kb_pdf (e.g. 'test1', 'root', 'all')")
    parser.add_argument("-c", "--collection", default="upsc_gs_v1", help="Chroma collection name")
    parser.add_argument("-r", "--reset", action="store_true", help="Wipe and recreate collection")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit number of PDFs processed")
    
    args = parser.parse_args()
    
    run_ingestion_pipeline(
        batch=args.batch,
        collection_name=args.collection,
        reset_collection=args.reset,
        limit=args.limit
    )
