from pathlib import Path
from typing import List
from langchain_core.documents import Document
from .parse import parse_pdf_to_markdown


def load_pdfs_from_directory(
    raw_dir: str = "kb_pdf",
    processed_dir: str = "data/kb_processed",
    recursive: bool = True,
    limit: int | None = None
) -> List[Document]:
    """
    Scans the raw directory (recursively or flat) for PDFs, parses them into Markdown,
    extracts simple metadata (source, doc_name, batch), and returns a list of Document objects.
    """
    raw_dir_obj = Path(raw_dir)
    documents = []
        
    if not raw_dir_obj.exists():
        print(f"Directory {raw_dir} does not exist. Creating it.")
        raw_dir_obj.mkdir(parents=True, exist_ok=True)
        return []
        
    # Find all PDF files
    if recursive:
        pdf_files = list(raw_dir_obj.rglob("*.pdf"))
    else:
        pdf_files = list(raw_dir_obj.glob("*.pdf"))
        
    if limit is not None and limit > 0:
        print(f"Limiting ingestion to the first {limit} file(s).")
        pdf_files = pdf_files[:limit]
        
    if not pdf_files:
        print(f"No PDF files found in {raw_dir} (recursive={recursive})")
        return []
        
    print(f"Found {len(pdf_files)} PDF file(s) to process.")
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            # Parse PDF to markdown (uses cache automatically if already parsed)
            markdown_content = parse_pdf_to_markdown(str(pdf_path), processed_dir)
            
            # Determine batch name (subfolder name)
            try:
                rel_path = pdf_path.relative_to(raw_dir_obj)
                if len(rel_path.parts) > 1:
                    batch = rel_path.parts[0]
                else:
                    batch = "root"
            except ValueError:
                batch = pdf_path.parent.name
            
            # Create LangChain Document with clean, direct metadata
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "source": pdf_path.name,
                    "doc_name": pdf_path.stem,  # e.g., 'NCERT-Class-12-Economics-Part-1'
                    "file_path": str(pdf_path),
                    "batch": batch,
                }
            )
            documents.append(doc)
            print(f"Loaded: {pdf_path.name} [Batch: {batch}]")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            
    return documents
