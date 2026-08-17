"""
Ingestion Pipeline (Phase 1: PDF Loading & Fixed-Size Chunking)

This script loads PDFs from the raw directory, parses them into Markdown,
and splits them into fixed-size chunks ready for subsequent embedding and vector storage.
"""

import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.loaders.pdf_loader import load_pdfs_from_directory, Document
from ingestion.chunking.fixed_size import chunk_documents


def run_ingestion_pipeline(
    raw_dir: str = "data/research_papers/raw",
    processed_dir: str = "data/research_papers/processed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Executes the ingestion pipeline up to the chunking phase.
    
    1. Loads raw PDFs and converts to Markdown Documents.
    2. Splits Document objects into fixed-size chunks with metadata.
    """
    print(f"=== Stage 1: Loading PDFs from '{raw_dir}' ===")
    documents = load_pdfs_from_directory(raw_dir=raw_dir, processed_dir=processed_dir)
    print(f"Successfully loaded {len(documents)} raw document(s).\n")

    if not documents:
        print("No documents found to process.")
        return []

    print(f"=== Stage 2: Fixed-Size Chunking (size={chunk_size}, overlap={chunk_overlap}) ===")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Successfully created {len(chunks)} total chunk(s).\n")

    # Print summary & preview of first chunk
    if chunks:
        print("--- Sample Chunk Preview ---")
        first_chunk = chunks[0]
        print(f"Metadata: {first_chunk.metadata}")
        print("Content Preview:")
        print(first_chunk.page_content[:300])
        print("----------------------------\n")

    return chunks


if __name__ == "__main__":
    run_ingestion_pipeline()
