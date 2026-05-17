from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = splitter.split_text(text)

    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]

    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks