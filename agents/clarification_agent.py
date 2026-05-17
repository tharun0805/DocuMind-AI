import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def needs_clarification(question: str) -> dict:
    logger.info(f"Checking clarification for: {question}")

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Analyze if this question needs clarification before answering.

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
            llm = get_shared_llm(temperature=0.2)
            chain = prompt | llm
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
                return {
                    "needs_clarification": True,
                    "questions": questions[:3]
                }

            return {"needs_clarification": False, "questions": []}

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return {"needs_clarification": False, "questions": []}

    return {"needs_clarification": False, "questions": []}