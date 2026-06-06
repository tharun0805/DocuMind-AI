from loguru import logger


def load_pdf(file_path: str) -> str:
    # PyMuPDF — 10x faster than pdfplumber, same quality
    try:
        import fitz
        doc  = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        logger.debug(f"PDF loaded via PyMuPDF: {len(text):,} chars")
        return text
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # pypdf — 3x faster fallback
    try:
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)
        logger.debug(f"PDF loaded via pypdf: {len(text):,} chars")
        return text
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")

    # pdfplumber — original (slow) last resort
    import pdfplumber
    with pdfplumber.open(file_path) as pdf:
        text = "".join(p.extract_text() or "" for p in pdf.pages)
    return text