import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.2
    )


def generate_answer(question: str, context: str) -> str:
    logger.info("Generating answer using QA agent...")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an intelligent document assistant.
        Answer the question using only the context provided below.
        If the answer is not in the context, say "I could not find this information in the document."
        Always be precise and cite which part of the context your answer comes from.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"context": context, "question": question})
            logger.info("Answer generated successfully")
            return result.content.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

    return "I could not generate an answer at this time. Please try again."