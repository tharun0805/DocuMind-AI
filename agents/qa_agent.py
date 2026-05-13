from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def generate_answer(question: str, context: str) -> str:
    logger.info("Generating answer using QA agent...")

    api_key = get_google_api_key()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

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

    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})

    logger.info("Answer generated successfully")
    return result.content.strip()