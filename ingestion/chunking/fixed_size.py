"""
Fixed-size and Character Chunking using LangChain Text Splitters.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_recursive: bool = False
) -> List[Document]:
    """
    Splits a list of LangChain Document objects into fixed-size chunks using LangChain text splitters,
    preserving existing metadata and enriching each chunk with tracking metadata.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlap length between adjacent chunks.
        use_recursive: If True, uses RecursiveCharacterTextSplitter; otherwise CharacterTextSplitter.

    Returns:
        List of chunked LangChain Document objects.
    """
    if use_recursive:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )

    chunked_docs = splitter.split_documents(documents)

    # Enrich metadata for tracking and citations
    for idx, doc in enumerate(chunked_docs):
        source_name = doc.metadata.get("source", "unknown")
        doc.metadata["chunk_id"] = f"{source_name}_chunk_{idx}"
        doc.metadata["chunk_index"] = idx
        doc.metadata["total_chunks"] = len(chunked_docs)
        doc.metadata["chunk_size"] = len(doc.page_content)

    return chunked_docs
