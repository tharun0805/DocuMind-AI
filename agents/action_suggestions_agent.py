import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def suggest_next_actions(question: str, answer: str) -> list[str]:
    logger.info("Generating next best actions...")

    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
        Based on this document question and answer suggest exactly 4
        next actions the user could take.

        Question: {question}
        Answer: {answer}

        Choose from: extract tasks, generate quiz, find contradictions,
        translate, create summary, draft email, rewrite for manager,
        find deadlines, extract key metrics, compare sections,
        identify risks, create outline.

        List exactly 4 short action labels 3-5 words each.
        One per line. No numbering. No bullets.

        Actions:
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            lines = [
                line.strip()
                for line in result.content.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]

            return lines[:4]

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return []

    return []