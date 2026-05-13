from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def classify_intent(question: str) -> str:
    logger.info(f"Classifying intent for: {question}")

    api_key = get_google_api_key()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )

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

    chain = prompt | llm
    result = chain.invoke({"question": question})
    intent = result.content.strip().lower()

    logger.info(f"Intent classified as: {intent}")
    return intent