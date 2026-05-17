import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def classify_intent(question: str) -> str:
    logger.info(f"Classifying intent for: {question}")

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Classify this question into exactly one category:
        - factual: specific information from document
        - computational: requires calculation or data analysis
        - summary: asking to summarize or overview

        Question: {question}

        Reply with only one word: factual, computational, or summary
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0)
            chain = prompt | llm
            result = chain.invoke({"question": question})
            intent = result.content.strip().lower()
            if intent not in ["factual", "computational", "summary"]:
                intent = "factual"
            logger.info(f"Intent: {intent}")
            return intent
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                logger.error(f"Intent error: {e}")
                return "factual"

    return "factual"