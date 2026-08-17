import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse, ResultType

# Load environment variables from the .env file (to get LLAMA_CLOUD_API_KEY)
load_dotenv()

def parse_pdf_to_markdown(pdf_path: str, processed_dir: str = "data/research_papers/processed") -> str:
    """
    Parses a single PDF using LlamaParse and caches the result locally as Markdown.
    If the cached markdown file already exists, it reads it directly to save API credits.
    """
    pdf_path_obj = Path(pdf_path)
    processed_dir_obj = Path(processed_dir)
    
    # 1. Ensure the processed/ directory exists
    processed_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # 2. Define the path where the cached Markdown version will live
    # Example: data/research_papers/raw/paper.pdf -> data/research_papers/processed/paper.md
    cache_path = processed_dir_obj / f"{pdf_path_obj.stem}.md"
    
    # 3. Check if we have a cache hit (file already parsed and saved)
    if cache_path.exists():
        print(f"[Cache Hit] Reading parsed content from local cache: {cache_path}")
        return cache_path.read_text(encoding="utf-8")
        
    # 4. Cache Miss: Call LlamaParse to convert the PDF to Markdown
    print(f"[Cache Miss] Parsing PDF via LlamaParse API: {pdf_path}")
    
    # Initialize the LlamaParse parser
    parser = LlamaParse(
        result_type=ResultType.MD,  # We want the output formatted as clean Markdown
    )
    
    # Parse the PDF (this makes a cloud network request to LlamaIndex servers)
    documents = parser.load_data(str(pdf_path_obj))
    
    # Reconstruct the full document text by joining all parsed pages/sections
    markdown_text = "\n\n".join([doc.text for doc in documents])
    
    # 5. Save the parsed result to our local cache folder
    cache_path.write_text(markdown_text, encoding="utf-8")
    print(f"[Parsed] Successfully cached parsed Markdown to: {cache_path}")
    
    return markdown_text
