import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


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

        Analyze all documents together.
        Compare information across documents when relevant.
        Indicate which document contains which information.
        Explain and summarize — never copy raw text.

        Answer:
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke({
                "question": question,
                "doc_contexts": doc_contexts
            })
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return "Could not query documents. Please try again."

    return "Could not query documents. Please try again."