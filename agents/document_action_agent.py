import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.4
    )


def perform_document_action(
    action: str,
    context: str,
    file_name: str = ""
) -> dict:
    logger.info(f"Performing document action: {action}")

    prompt = PromptTemplate(
        input_variables=["action", "context", "file_name"],
        template="""
        You are DocuMind AI — an expert document transformation assistant.

        Document: {file_name}

        Document Content:
        {context}

        Requested Action: {action}

        Perform this action thoroughly and professionally.
        Format your response clearly with proper structure.
        Explain and elaborate — do not just copy from the document.
        Provide maximum value and depth in your output.

        Result:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "action": action,
                "context": context,
                "file_name": file_name
            })

            output = result.content.strip()
            logger.info("Document action completed successfully")

            return {
                "success": True,
                "result": output,
                "action": action
            }

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {
        "success": False,
        "result": "Could not perform action. Please try again.",
        "action": action
    }