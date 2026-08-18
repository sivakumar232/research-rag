"""
Ingestion Pipeline (Phase 1: PDF Loading, Chunking, Embedding, & Storage)

This script loads PDFs from the raw directory, parses them into Markdown,
splits them into fixed-size chunks, computes embeddings, and stores them in ChromaDB.
"""

print("🚀 Starting Ingestion Script... ")

import sys
import uuid
from pathlib import Path
from typing import List
import chromadb

# Ensure project root is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.loaders.pdf_loader import load_pdfs_from_directory, Document
from ingestion.chunking.fixed_size import chunk_documents
from ingestion.embedding.embedder import get_embedding_model


def run_ingestion_pipeline(
    raw_dir: str = "data/research_papers/raw",
    processed_dir: str = "data/research_papers/processed",
    chroma_db_dir: str = "data/chroma_db",
    collection_name: str = "research_papers_v1",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Executes the ingestion pipeline.
    
    1. Loads raw PDFs and converts to Markdown Documents.
    2. Splits Document objects into fixed-size chunks.
    3. Computes vector embeddings for each chunk.
    4. Stores text, metadata, and embeddings in ChromaDB.
    """
    print(f"=== Stage 1: Loading PDFs from '{raw_dir}' ===")
    documents = load_pdfs_from_directory(raw_dir=raw_dir, processed_dir=processed_dir)
    print(f"Successfully loaded {len(documents)} raw document(s).\n")

    if not documents:
        print("No documents found to process.")
        return

    print(f"=== Stage 2: Fixed-Size Chunking (size={chunk_size}, overlap={chunk_overlap}) ===")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Successfully created {len(chunks)} total chunk(s).\n")

    if not chunks:
        print("No chunks created.")
        return

    print("=== Stage 3: Computing Embeddings & Storing in ChromaDB ===")
    
    # Initialize embedding model
    embedder = get_embedding_model()
    
    # Initialize ChromaDB client (Cloud / Remote)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")
    
    print(f"Connecting to ChromaDB Cloud (Tenant: {chroma_tenant}, Database: {chroma_database})...")
    chroma_client = chromadb.CloudClient(
        tenant=chroma_tenant,
        database=chroma_database,
        api_key=chroma_api_key
    )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Prepare data for ChromaDB
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Compute embeddings
    print("Computing embeddings... (this may take a moment)")
    embeddings = embedder.embed_documents(texts)
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    print(f"Upserting {len(chunks)} chunks into ChromaDB collection '{collection_name}' in batches...")
    
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        end_idx = i + batch_size
        print(f"Upserting batch {i//batch_size + 1} (items {i} to {min(end_idx, len(chunks))})...")
        collection.upsert(
            ids=ids[i:end_idx],
            documents=texts[i:end_idx],
            embeddings=embeddings[i:end_idx],  # type: ignore
            metadatas=metadatas[i:end_idx]     # type: ignore
        )
    
    print(f"Success! Embedded and stored {len(chunks)} chunks in {chroma_db_dir}.\n")
    
    # Verify by querying the collection
    count = collection.count()
    print(f"Total documents in collection '{collection_name}': {count}")


if __name__ == "__main__":
    run_ingestion_pipeline()
