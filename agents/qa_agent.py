import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def generate_answer(
    question: str,
    context: str,
    chat_history: str = "",
    answer_mode: str = "detailed"
) -> dict:
    logger.info(f"Generating answer in mode: {answer_mode}")

    mode_instructions = {
        "detailed": "Give a thorough well structured detailed answer. Explain concepts clearly and make sure the user fully understands.",
        "quick": "Give a short direct answer in 2-3 sentences maximum.",
        "bullet": "Give the answer as clear concise bullet points only.",
        "beginner": "Explain in very simple language. Use analogies and avoid jargon.",
        "executive": "Give a brief executive summary focused on key decisions and outcomes.",
        "table": "Present the answer as a structured markdown table where possible."
    }

    mode_text = mode_instructions.get(answer_mode, mode_instructions["detailed"])

    base_template = """
    You are DocuMind AI — an expert document analyst and intelligent assistant.

    Your job is to:
    - UNDERSTAND the document content deeply
    - EXPLAIN it clearly in your own words
    - SUMMARIZE complex information simply
    - CONNECT related pieces of information together
    - ANSWER general questions by relating them to the document context
    - If exact information is not in the document use the context to give a helpful general answer

    Output style: {mode_text}

    Rules:
    - Never paste raw text from the document
    - Always explain in clear natural language
    - If exact answer not found say: "This is not directly covered in the document, but based on the content:"
    - Always end with: Key Takeaway: [one clear sentence]
    """

    if chat_history:
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history", "mode_text"],
            template=base_template + """
            Previous Conversation:
            {chat_history}

            Document Content:
            {context}

            Question: {question}

            Answer:
            """
        )
        inputs = {
            "context": context,
            "question": question,
            "chat_history": chat_history,
            "mode_text": mode_text
        }
    else:
        prompt = PromptTemplate(
            input_variables=["context", "question", "mode_text"],
            template=base_template + """
            Document Content:
            {context}

            Question: {question}

            Answer:
            """
        )
        inputs = {
            "context": context,
            "question": question,
            "mode_text": mode_text
        }

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke(inputs)
            answer_text = result.content.strip()
            logger.info("Answer generated successfully")
            chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
            evidence = chunks[:3]
            return {
                "answer": answer_text,
                "evidence": evidence,
                "mode": answer_mode
            }
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {
        "answer": "Could not generate answer. Please try again.",
        "evidence": [],
        "mode": answer_mode
    }