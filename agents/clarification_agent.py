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


def needs_clarification(question: str) -> dict:
    logger.info(f"Checking clarification for: {question}")

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Analyze if this question needs clarification before answering.
        Consider: target audience, format, scope, depth level, time range.

        Question: {question}

        If clarification would significantly improve the answer respond:
        NEEDS_CLARIFICATION: yes
        QUESTIONS:
        1. [first clarifying question]
        2. [second clarifying question]

        If question is clear enough respond:
        NEEDS_CLARIFICATION: no

        Response:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"question": question})
            content = result.content.strip()

            if "NEEDS_CLARIFICATION: yes" in content:
                lines = content.split("\n")
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit() and "." in line:
                        q = line.split(".", 1)[1].strip()
                        if q:
                            questions.append(q)
                logger.info(f"Clarification needed: {len(questions)} questions")
                return {
                    "needs_clarification": True,
                    "questions": questions[:3]
                }

            return {"needs_clarification": False, "questions": []}

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {"needs_clarification": False, "questions": []}