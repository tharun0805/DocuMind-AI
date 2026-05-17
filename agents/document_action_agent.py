import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def perform_document_action(
    action: str,
    context: str,
    file_name: str = ""
) -> dict:
    logger.info(f"Performing action: {action}")

    prompt = PromptTemplate(
        input_variables=["action", "context", "file_name"],
        template="""
        You are DocuMind AI — an expert document transformation assistant.

        Document: {file_name}

        Document Content:
        {context}

        Requested Action: {action}

        Perform this action thoroughly and professionally.
        Explain and elaborate — do not just copy from document.

        Result:
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.4)
            chain = prompt | llm
            result = chain.invoke({
                "action": action,
                "context": context,
                "file_name": file_name
            })
            return {
                "success": True,
                "result": result.content.strip(),
                "action": action
            }

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return {
                    "success": False,
                    "result": "Could not perform action. Please try again.",
                    "action": action
                }

    return {
        "success": False,
        "result": "Could not perform action. Please try again.",
        "action": action
    }