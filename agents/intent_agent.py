from loguru import logger


def classify_intent(question: str) -> str:
    """
    Classify question intent purely by question pattern.
    NO hardcoded domain terms. Works for ANY document type.
    """
    logger.debug(f"Classifying intent: {question[:50]}")
    ql = question.lower()

    computational = [
        "calculate", "compute", "sum", "total", "average", "count",
        "maximum", "minimum", "how many", "percentage", "ratio",
        "how much", "mean", "median", "std", "correlation",
        "highest", "lowest", "top", "bottom", "rank",
        "greater than", "less than", "most", "least",
        "score total", "total score", "sum of",
        "oldest", "youngest", "largest", "smallest",
        "compare", "versus",
    ]
    summary = [
        "summarize", "summary", "overview", "brief", "outline",
        "main points", "key points", "what is this about", "describe",
        "identify risks", "list risks", "what are the risks",
        "identify issues", "key findings", "generate questions",
        "create questions", "explain", "tell me about",
        "what does this", "analyse", "analyze", "interpret",
    ]

    for kw in computational:
        if kw in ql:
            return "computational"
    for kw in summary:
        if kw in ql:
            return "summary"
    return "factual"