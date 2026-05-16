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
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            lines = [
                line.strip()
                for line in result.content.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]

            actions = lines[:4]
            logger.info(f"Generated {len(actions)} next actions")
            return actions

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return []
