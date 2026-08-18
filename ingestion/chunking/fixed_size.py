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

    chunked_docs = []
    for doc in documents:
        # Split each document individually to calculate proper per-document metadata
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            source_name = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_id"] = f"{source_name}_chunk_{idx}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(doc_chunks)
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunked_docs.append(chunk)

    return chunked_docs
