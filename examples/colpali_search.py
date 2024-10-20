from searchagent.colpali.base import ColPaliRag
from searchagent.utils import project_paths

pdfs_dir = project_paths.PDF_DIR
single_file_dir = project_paths.SINGLE_FILE_DIR
embeddings_filepath = f"{project_paths}/embeddings_metadata.json"

# Page 4 of the paper Attention Is All You Need
rag = ColPaliRag(input_dir=pdfs_dir, store_locally=False)
rag.search(query="Find a page on 'Scaled Dot-Product Attention'")
