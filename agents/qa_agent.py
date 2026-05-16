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


def generate_answer(
    question: str,
    context: str,
    chat_history: str = "",
    answer_mode: str = "detailed"
) -> dict:
    logger.info(f"Generating answer in mode: {answer_mode}")

    mode_instructions = {
        "detailed": "Give a thorough well structured detailed answer. Explain concepts clearly, provide context, and make sure the user fully understands the topic.",
        "quick": "Give a short direct answer in 2-3 sentences maximum. Focus only on the most important point.",
        "bullet": "Give the answer as clear concise bullet points only. Each bullet should be a complete meaningful point.",
        "beginner": "Explain in very simple language as if explaining to a complete beginner. Use analogies, simple words, avoid jargon.",
        "executive": "Give a brief executive summary focused on key decisions, outcomes, risks and recommended actions.",
        "table": "Present the answer as a structured markdown table where possible."
    }

    mode_text = mode_instructions.get(
        answer_mode,
        mode_instructions["detailed"]
    )

    base_template = """
    You are DocuMind AI — an expert document analyst and intelligent assistant.

    Your job is to:
    - UNDERSTAND the document content deeply
    - EXPLAIN it clearly in your own words
    - SUMMARIZE complex information simply
    - CONNECT related pieces of information together
    - HIGHLIGHT what is most important and relevant
    - Answer as an expert who has fully read and understood the document

    Output style: {mode_text}

    Rules:
    - Never paste raw text from the document
    - Always explain in clear natural language
    - If something is complex break it down simply
    - If the answer is not in the document say exactly:
      I could not find this information in the document.
    - Always end with a one line key takeaway prefixed with Key Takeaway:
    """

    if chat_history:
        prompt = PromptTemplate(
            input_variables=[
                "context", "question", "chat_history", "mode_text"
            ],
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
            chain = prompt | get_llm()
            result = chain.invoke(inputs)
            answer_text = result.content.strip()
            logger.info("Answer generated successfully")

            chunks = [
                c.strip()
                for c in context.split("\n\n")
                if c.strip()
            ]
            evidence = chunks[:3]

            return {
                "answer": answer_text,
                "evidence": evidence,
                "mode": answer_mode
            }

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                raise e

    return {
        "answer": "I could not generate an answer. Please try again.",
        "evidence": [],
        "mode": answer_mode
    }