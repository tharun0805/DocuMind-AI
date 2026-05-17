from loguru import logger


def classify_intent(question: str) -> str:
    logger.info(f"Classifying intent for: {question}")

    question_lower = question.lower()

    computational_keywords = [
        "calculate", "compute", "sum", "total", "average", "count",
        "maximum", "minimum", "how many", "percentage", "ratio",
        "how much", "add up", "subtract", "multiply", "divide"
    ]

    summary_keywords = [
        "summarize", "summary", "overview", "brief", "outline",
        "main points", "key points", "what is this about", "describe",
        "what does this document"
    ]

    for keyword in computational_keywords:
        if keyword in question_lower:
            logger.info("Intent: computational")
            return "computational"

    for keyword in summary_keywords:
        if keyword in question_lower:
            logger.info("Intent: summary")
            return "summary"

    logger.info("Intent: factual")
    return "factual"