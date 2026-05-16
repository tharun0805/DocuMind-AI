import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.1
    )


def generate_preview(question: str, context: str) -> str:
    logger.info("Generating fast preview response...")

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        Give a very quick 1-2 sentence preview answer to this question.
        This is a fast preview before the full detailed answer.

        Context: {context}
        Question: {question}

        Quick Preview 1-2 sentences only:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "context": context[:1000]
            })

            preview = result.content.strip()
            logger.info("Preview generated successfully")
            return preview

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return ""