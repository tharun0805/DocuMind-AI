import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.3
    )


def query_multiple_documents(
    question: str,
    documents: list[dict]
) -> str:
    logger.info(f"Querying {len(documents)} documents...")

    doc_contexts = ""
    for i, doc in enumerate(documents, 1):
        doc_contexts += f"\n--- Document {i}: {doc['name']} ---\n"
        doc_contexts += doc["text"][:2000]
        doc_contexts += "\n"

    prompt = PromptTemplate(
        input_variables=["question", "doc_contexts"],
        template="""
        You are DocuMind AI analyzing multiple documents simultaneously.

        Documents:
        {doc_contexts}

        Question: {question}

        Answer by:
        - Analyzing all documents together
        - Comparing information across documents when relevant
        - Clearly indicating which document contains which information
        - Providing a comprehensive synthesized answer
        - Explaining and summarizing — never copying raw text

        Answer:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "doc_contexts": doc_contexts
            })

            logger.info("Multi-document query complete")
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return "Could not query documents. Please try again."