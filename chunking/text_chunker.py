from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks