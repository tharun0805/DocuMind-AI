import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def generate_preview(question: str, context: str) -> str:
    logger.info("Generating fast preview response...")

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        Give a very quick 1-2 sentence preview answer to this question.

        Context: {context}
        Question: {question}

        Quick Preview 1-2 sentences only:
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.1)
            chain = prompt | llm
            result = chain.invoke({
                "question": question,
                "context": context[:1000]
            })
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return ""

    return ""