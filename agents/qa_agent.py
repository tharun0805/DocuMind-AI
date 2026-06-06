import time
from langchain_core.prompts import PromptTemplate
from loguru import logger
from utils.llm_provider import get_shared_llm


def generate_answer(
    question: str,
    context: str,
    chat_history: str = "",
    answer_mode: str = "detailed",
    doc_text: str = "",
) -> dict:
    logger.debug(f"QA: mode={answer_mode} q={question[:60]}")

    mode_instructions = {
        "detailed":  "Write a thorough, well-structured answer with all relevant details.",
        "quick":     "Answer in 2-4 sentences. Direct and specific.",
        "bullet":    "Clear bullet points with specific facts from the document.",
        "beginner":  "Simple everyday language. No jargon. Use analogies if helpful.",
        "executive": "Key facts only. Brief, direct, actionable.",
        "table":     "Format as a clean markdown table where data supports it.",
    }
    mode_text = mode_instructions.get(answer_mode, mode_instructions["detailed"])

    effective_context = context or ""
    if doc_text and len(effective_context.strip()) < 300:
        effective_context = doc_text[:7000]
    elif doc_text and len(effective_context) < 1500:
        effective_context = effective_context + "\n\n" + doc_text[:2500]

    history_block = (
        f"CONVERSATION HISTORY:\n{chat_history}\n\n" if chat_history else ""
    )

    ql = question.lower()
    is_followup = any(p in ql for p in [
        "you said", "you told", "you provided", "previously", "earlier",
        "follow up", "more about", "elaborate", "expand on",
        "what about", "also tell", "continue",
    ])
    followup_note = (
        "\nThis is a follow-up — build on the previous answer, do not repeat it.\n"
        if is_followup else ""
    )

    # Rule 4 is intentionally generic: the LLM must read the document's own
    # scoring ranges and labels — never assume any fixed classification system.
    tpl = """You are DocuMind AI — a highly intelligent, domain-agnostic document analyst.
You work with ANY type of document: books, reports, surveys, legal, financial, scientific, etc.

RETRIEVED DOCUMENT CONTENT:
{context}

{history_block}{followup_note}QUESTION: {question}

RESPONSE STYLE: {mode_text}

RULES:
1. Answer using ONLY the retrieved content above — never hallucinate
2. Be specific — use actual facts, names, numbers, and concepts from the document
3. Adapt intelligently to the document type:
   • Book/article → key concepts, arguments, examples, actionable insights
   • Survey/questionnaire → patterns, totals, comparisons, findings
   • Legal/financial → terms, clauses, figures, implications
   • Scientific → methodology, findings, conclusions, significance
4. For any scored or assessed data: read the scoring system from the document
   itself — use the ranges, labels, and categories the document defines,
   never assume a fixed classification system
5. Structure clearly — use sections/bullets only when it aids comprehension
6. Never add "Introduction" or "Conclusion" headers unless asked for a report
7. For follow-ups: build on prior context, never repeat yourself
8. If content is insufficient, say what IS known and what needs clarification
9. Match length to complexity — simple question = concise answer

ANSWER:"""

    for attempt in range(3):
        try:
            llm    = get_shared_llm(temperature=0.1)
            prompt = PromptTemplate(
                input_variables=["context", "question", "mode_text",
                                 "history_block", "followup_note"],
                template=tpl,
            )
            result = (prompt | llm).invoke({
                "context":       effective_context[:7000],
                "question":      question,
                "mode_text":     mode_text,
                "history_block": history_block,
                "followup_note": followup_note,
            })
            answer = (result.content.strip()
                      if hasattr(result, "content") else str(result).strip())
            if not answer:
                raise ValueError("empty")

            chunks   = effective_context.split("\n\n---\n\n")
            evidence = [c.strip() for c in chunks if len(c.strip()) > 80][:4]
            return {"answer": answer, "evidence": evidence, "mode": answer_mode}

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"QA error: {e}")
                if attempt == 2:
                    break

    return {
        "answer": "The AI service is temporarily busy. Please try again.",
        "evidence": [],
        "mode": answer_mode,
    }