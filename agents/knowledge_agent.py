import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def expand_knowledge(question: str, answer: str) -> dict:
    logger.info("Expanding knowledge...")

    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
        Based on this question and answer suggest learning resources.

        Question: {question}
        Answer: {answer}

        YOUTUBE_SEARCHES: [3 YouTube search queries one per line]

        RELATED_TOPICS: [3 related topics one per line]

        SIMILAR_RESOURCES: [3 resource types one per line]

        LEARN_MORE: [2 things to study next one per line]
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

            content = result.content.strip()
            knowledge = {}
            current_key = None
            current_items = []

            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue

                found = False
                for key in [
                    "YOUTUBE_SEARCHES", "RELATED_TOPICS",
                    "SIMILAR_RESOURCES", "LEARN_MORE"
                ]:
                    if line.startswith(key + ":"):
                        if current_key:
                            knowledge[current_key] = current_items
                        current_key = key
                        rest = line[len(key)+1:].strip()
                        current_items = [rest] if rest else []
                        found = True
                        break

                if not found and current_key and line:
                    current_items.append(line)

            if current_key:
                knowledge[current_key] = current_items

            return knowledge

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return {}

    return {}