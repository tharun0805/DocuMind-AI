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
        "detailed":  "Give a focused, well-structured answer with all relevant details.",
        "quick":     "Answer in 2-3 sentences maximum. Be direct and specific.",
        "bullet":    "Answer as clear, specific bullet points only.",
        "beginner":  "Explain simply in everyday language. No jargon.",
        "executive": "Key facts and decisions only. Brief and actionable.",
        "table":     "Format as a markdown table where data supports it.",
    }
    mode_text = mode_instructions.get(answer_mode, mode_instructions["detailed"])
 
    effective_context = context or ""
    if doc_text and len(effective_context.strip()) < 300:
        effective_context = doc_text[:7000]
    elif doc_text and len(effective_context) < 1500:
        effective_context = effective_context + "\n\n[Additional context:]\n" + doc_text[:3000]
 
    history_section = f"Conversation History:\n{chat_history}\n\n" if chat_history else ""
 
    tpl = """You are DocuMind AI — a precise, expert document analyst.
You work with ANY document type: books, reports, surveys, financial, legal, scientific, etc.
 
DOCUMENT CONTENT:
{context}
 
{history_section}QUESTION: {question}
 
RESPONSE STYLE: {mode_text}
 
RULES:
1. Answer ONLY what the question asks — no unrelated sections
2. NEVER write "Introduction", "Background", or "Overview" unless asked
3. NEVER show step-by-step calculations unless the user says "show working"
4. NEVER reproduce raw data verbatim — CSV rows and pipe-separated values
   must always be interpreted and summarised in plain English
5. For any scoring or assessment data: detect the scoring system from the
   document itself, compute totals, and classify results using whatever
   ranges or labels the document defines — never assume a fixed scoring system
6. For "identify risks" give a concise named list of actual risk items only
7. For "generate questions" output numbered questions only, no preamble
8. Quote specific names, values, dates from the document when relevant
9. If information is not in the document, say so clearly — never guess
 
ANSWER:"""
 
    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.1)
            prompt = PromptTemplate(
                input_variables=["context", "question", "mode_text", "history_section"],
                template=tpl,
            )
            result = (prompt | llm).invoke({
                "context":         effective_context[:7000],
                "question":        question,
                "mode_text":       mode_text,
                "history_section": history_section,
            })
            answer = result.content.strip() if hasattr(result, "content") else str(result).strip()
            if not answer:
                raise ValueError("empty response")
 
            raw_chunks = effective_context.split("\n\n---\n\n")
            evidence   = [c.strip() for c in raw_chunks if len(c.strip()) > 80][:4]
            return {"answer": answer, "evidence": evidence, "mode": answer_mode}
 
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                logger.warning(f"Rate limit attempt {attempt+1}, waiting {30*(attempt+1)}s")
                time.sleep(30 * (attempt + 1))
            else:
                logger.error(f"QA agent error: {e}")
                if attempt == 2:
                    break
 
    return {
        "answer": "The AI service is temporarily busy. Please try again in a moment.",
        "evidence": [],
        "mode": answer_mode,
    }