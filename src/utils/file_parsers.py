# src/utils/file_parsers.py
import os
from typing import Generator, List

def stream_jsonl_chunks(file_path: str, chunk_size: int = 100) -> Generator[List[str], None, None]:
    """
    Memory-bounded stream generator. Reads a massive JSONL file line-by-line,
    batching rows into discrete chunks to protect local CPU memory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file not found at path: {file_path}")
        
    chunk = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            chunk.append(clean_line)
            
            # Once the chunk size limit is reached, yield it and clear memory
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
                
        # Yield any remaining lines left at the end of the file
        if chunk:
            yield chunk