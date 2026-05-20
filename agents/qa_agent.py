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
        "detailed": "Give a thorough, well-structured, detailed answer. Explain concepts clearly with examples.",
        "quick": "Give a short direct answer in 2-3 sentences. Focus on the key point only.",
        "bullet": "Give the answer as clear concise bullet points. Each point should be meaningful.",
        "beginner": "Explain in very simple language a beginner can understand. Use analogies and simple words.",
        "executive": "Give a brief executive summary. Focus on key decisions, outcomes and recommendations.",
        "table": "Present the answer as a structured markdown table where appropriate."
    }

    mode_text = mode_instructions.get(answer_mode, mode_instructions["detailed"])

    base = """
You are DocuMind AI - an expert document analyst and intelligent assistant.

CRITICAL RULES:
- You MUST always provide a helpful answer
- If the exact information is not in the document context, use the context to give the BEST POSSIBLE related answer
- NEVER say "I could not find this information" - always explain what IS in the document instead
- NEVER copy raw text - always explain in your own clear words
- ALWAYS summarize and explain, not copy-paste
- Connect related pieces of information intelligently
- End EVERY answer with: "Key Takeaway: [one clear sentence]"

Output style: {mode_text}
"""

    if chat_history:
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history", "mode_text"],
            template=base + """
Previous Conversation:
{chat_history}

Document Content:
{context}

Question: {question}

Answer:"""
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
            template=base + """
Document Content:
{context}

Question: {question}

Answer:"""
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
        "answer": "I encountered an error generating the answer. Please try again.",
        "evidence": [],
        "mode": answer_mode
    }
