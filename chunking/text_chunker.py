from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    chunk_size: int = 2000,   # increased from 1500 → fewer chunks → faster FAISS build
    chunk_overlap: int = 150,  # slightly larger overlap keeps context quality
    max_chunks: int = 200,     # reduced cap: 300 chunks was overkill for most docs
) -> list[str]:
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    if len(chunks) > max_chunks:
        step   = max(1, len(chunks) // max_chunks)
        chunks = chunks[::step][:max_chunks]

    logger.debug(f"Chunked: {len(chunks)} chunks from {len(text):,} chars")
    return chunks