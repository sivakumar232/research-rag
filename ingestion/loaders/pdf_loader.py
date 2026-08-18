from pathlib import Path
from typing import List
from langchain_core.documents import Document
from .parse import parse_pdf_to_markdown


def load_pdfs_from_directory(
    raw_dir: str = "data/research_papers/raw",
    processed_dir: str = "data/research_papers/processed"
) -> List[Document]:
    """
    Scans the raw directory for PDFs, parses them into Markdown,
    and returns a list of Document objects.
    """
    raw_dir_obj = Path(raw_dir)
    documents = []
        
    # 1. Check if the raw directory exists
    if not raw_dir_obj.exists():
        print(f"Directory {raw_dir} does not exist. Creating it.")
        raw_dir_obj.mkdir(parents=True, exist_ok=True)
        return []
        
    # 2. Find all PDF files in the raw folder
    pdf_files = list(raw_dir_obj.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return []
        
    print(f"Found {len(pdf_files)} PDF file(s) to process.")
    
    # 3. Loop through each PDF and parse it
    for pdf_path in pdf_files:
        try:
            # Parse the PDF (uses cache automatically if already parsed)
            markdown_content = parse_pdf_to_markdown(str(pdf_path), processed_dir)
            
            # Wrap the parsed text and metadata into a standard Document object
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "source": pdf_path.name,
                    "file_path": str(pdf_path),
                }
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            
    return documents
