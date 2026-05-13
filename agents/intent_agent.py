import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0
    )


def classify_intent(question: str) -> str:
    logger.info(f"Classifying intent for: {question}")

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Classify the following question into exactly one of these categories:
        - factual: questions about specific information in a document
        - computational: questions that require calculation or data analysis
        - summary: questions asking to summarize or overview content

        Question: {question}

        Reply with only one word: factual, computational, or summary
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"question": question})
            intent = result.content.strip().lower()
            logger.info(f"Intent classified as: {intent}")
            return intent
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

    return "factual"