import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import time, io, re, json
from loguru import logger

# â”€â”€ Silence terminal noise â”€â”€
import sys as _sys
logger.remove()
logger.add(_sys.stderr, level="WARNING")
# Only add file logger if logs dir exists (don't block startup creating dirs)
import os as _os
if _os.path.isdir("logs") or not _os.path.exists("logs"):
    try:
        _os.makedirs("logs", exist_ok=True)
        logger.add("logs/documind.log", level="DEBUG", rotation="10 MB", retention="7 days", enqueue=True)
    except Exception:
        pass

st.set_page_config(page_title="DocuMind AI", page_icon="DM", layout="wide", initial_sidebar_state="expanded")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CORE â€” cached once, never reloaded
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@st.cache_resource(show_spinner=False)
def load_core():
    from ingestion.document_loader import load_document
    from chunking.text_chunker import chunk_text
    from vector_store.faiss_store import create_vector_store
    from vector_store.bm25_store import create_bm25_index
    from graph.workflow import run_workflow
    from memory.session_memory import SessionMemory
    from memory.file_memory_manager import FileMemoryManager
    from utils.validator import validate_file, validate_question
    return dict(load_document=load_document, chunk_text=chunk_text,
                create_vector_store=create_vector_store, create_bm25_index=create_bm25_index,
                run_workflow=run_workflow, SessionMemory=SessionMemory,
                FileMemoryManager=FileMemoryManager, validate_file=validate_file,
                validate_question=validate_question)

@st.cache_resource(show_spinner=False)
def _llm_groq():
    key = os.getenv("GROQ_API_KEY", "")
    if key and key not in ["your_groq_key_here", ""]:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=key, temperature=0.3, max_tokens=2048)
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def _llm_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from utils.config import get_google_api_key
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=get_google_api_key(), temperature=0.3)

def get_llm(temp=0.3):
    llm = _llm_groq() or _llm_gemini()
    if temp != 0.3:
        try:
            return llm.with_config(configurable={"temperature": temp})
        except Exception:
            pass
    return llm

def lazy(mod, fn):
    import importlib
    return getattr(importlib.import_module(mod), fn)

def uid():
    return f"{int(time.time()*1000)}_{len(st.session_state.get('chat', []))}"

def init(deps):
    simple = {"chat": [], "file_path": "", "doc_ready": False, "doc_name": "", "doc_text": "",
              "prompts": [], "answer_mode": "detailed", "multi_docs": [], "last_q": "", "last_a": "",
              "pending": {}, "quiz_store": {}, "_processing": False, "_stop_flag": False}
    for k, v in simple.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "memory" not in st.session_state:
        st.session_state["memory"] = deps["SessionMemory"]()
    if "fmm" not in st.session_state:
        st.session_state["fmm"] = deps["FileMemoryManager"]()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# INTENT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def detect_intent(q):
    ql = q.lower()
    return {
        "chart":      any(w in ql for w in ["chart","graph","plot","bar chart","pie chart","line graph","visualize","histogram","draw a","show me a","bar graph","visual","figures","graphically","scatter","distribution"]),
        "quiz":       any(w in ql for w in ["quiz","test me","mcq","multiple choice","exam","question me","conduct a quiz","give me a quiz","ask me questions"]),
        "excel":      any(w in ql for w in ["excel","xlsx","generate excel","export excel","spreadsheet file"]),
        "pptx":       any(w in ql for w in ["powerpoint","pptx","presentation","make slides","create slides"]),
        "export_doc": any(w in ql for w in ["export as","save as","generate pdf","create pdf","export pdf","generate doc","export doc","generate csv","download as"]),
        "resources":  any(w in ql for w in ["resources","youtube links","learn more","reference links","study material","more information","show resources"]),
        "regen_doc":  any(w in ql for w in ["regenerate","shorten doc","rewrite doc","create document","generate document","make document","shorten to","regenerate document","create doc","make a doc"]),
    }

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LLM CALL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def llm_call(tpl, vars_, temp=0.3):
    from langchain_core.prompts import PromptTemplate
    r = (PromptTemplate(input_variables=list(vars_.keys()), template=tpl) | get_llm(temp)).invoke(vars_)
    return (r.content if hasattr(r, "content") else str(r)).strip()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DIRECT SUMMARIZE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def direct_summarize(doc_text, action, mode="detailed"):
    if not doc_text or not doc_text.strip():
        return "No document content. Please upload and process a document first."
    mode_map = {
        "detailed":  "Give a thorough structured answer with all key details.",
        "quick":     "Give a direct answer in 2-3 sentences only.",
        "bullet":    "Give the answer as clear specific bullet points.",
        "beginner":  "Explain in simple terms a beginner can understand.",
        "executive": "Give a brief executive summary focused on key decisions.",
        "table":     "Present information as a markdown table where appropriate.",
    }
    tpl = """You are DocuMind AI â€” an expert AI document analyst with broad general knowledge.

Document Content:
{doc_text}

Task: {action}
Response Style: {mode_text}

RULES:
1. Answer primarily from the document content above
2. Use your general knowledge to ENRICH and EXPLAIN (e.g. explain what BDI/BAI means, provide context)
3. For tabular/data documents: interpret numbers meaningfully in plain English
4. For summaries: describe what the data represents, who it covers, key patterns
5. Be specific â€” cite actual values, names, dates from the document
6. Structure your response with headers/bullets where appropriate
7. If something is not in the document, say what IS there and supplement with general knowledge
8. End with: **Key Takeaway:** [one precise, useful sentence]

Answer:"""
    try:
        ans = llm_call(tpl, {"doc_text": doc_text[:12000], "action": action,
                              "mode_text": mode_map.get(mode, mode_map["detailed"])})
        if not ans or ans.lower().strip() in ["none", "result: none", ""]:
            raise ValueError("empty")
        return ans
    except Exception as e:
        logger.error(f"direct_summarize: {e}")
        lines = [l.strip() for l in doc_text.split("\n") if l.strip()]
        return "**Document Preview:**\n\n" + "\n".join(lines[:25]) + "\n\n_Try rephrasing._"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DOCUMENT REGENERATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def regen_document(doc_text, instruction):
    tpl = """Transform this document per the instruction.

Original Document:
{doc_text}

Instruction: {instruction}

RULES:
- Follow PRECISELY (e.g. "10 records" = exactly 10 data rows)
- Keep ALL original column/field headers
- Output ONLY the transformed content - no preamble, no explanation
- For tabular data: keep exact same format as original

Output:"""
    try:
        new_content = llm_call(tpl, {"doc_text": doc_text[:10000], "instruction": instruction}, temp=0.1)
    except Exception as e:
        logger.error(f"regen_document: {e}")
        new_content = doc_text[:3000]
    return {"docx": gen_file(new_content, "docx"), "pdf": gen_file(new_content, "pdf"),
            "csv": gen_file(new_content, "csv"), "txt": gen_file(new_content, "txt")}, new_content

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _make_fig(labels, values, title, ctype):
    try:
        import plotly.graph_objects as go
        C = ["#6366f1","#06b6d4","#8b5cf6","#10b981","#f59e0b","#ef4444","#ec4899","#14b8a6"]
        L = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                 font=dict(color="#94a3b8", size=13),
                 title=dict(text=title, font=dict(color="#e2e8f0", size=16), x=0.5),
                 margin=dict(t=60, b=50, l=50, r=30), height=380, showlegend=(ctype == "pie"))
        if ctype == "pie":
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4,
                                   marker=dict(colors=C[:len(labels)]), textinfo="percent+label"))
        elif ctype == "line":
            fig = go.Figure(go.Scatter(x=labels, y=values, mode="lines+markers",
                                       line=dict(color="#6366f1", width=3),
                                       marker=dict(size=8, color="#06b6d4"),
                                       fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            L.update(xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        else:
            fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=C[:len(labels)])))
            L.update(xaxis=dict(gridcolor="rgba(255,255,255,0.03)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), bargap=0.3)
        fig.update_layout(**L)
        return fig
    except Exception:
        return None

def gen_charts(doc_text, question):
    tpl = """You are a data visualization expert. Extract numerical data to create charts for: {question}

Document/Data:
{text}

Instructions:
- Find actual numerical values in the document
- For tabular data (CSV/Excel): use column values for meaningful charts
- Create up to 3 different chart types showing different insights
- Return ONLY valid JSON array, no markdown, no explanation:

[{{"labels":["Category A","Category B","Category C"],"values":[25,40,35],"title":"Descriptive Chart Title","type":"bar"}}]

Chart types: bar, pie, line
Ensure labels and values arrays have the same length.
If no clear numerical data exists, infer approximate distributions from text."""
    try:
        raw = llm_call(tpl, {"text": doc_text[:3000], "question": question}, temp=0)
        raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s < 0: return []
        figs = []
        for cd in json.loads(raw[s:e])[:3]:
            lb, vl = cd.get("labels", []), cd.get("values", [])
            if lb and vl and len(lb) == len(vl):
                f = _make_fig(lb, [float(v) for v in vl], cd.get("title","Chart"), cd.get("type","bar"))
                if f: figs.append((cd.get("title","Chart"), f))
        return figs
    except Exception as e:
        logger.error(f"gen_charts: {e}")
        return []

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# QUIZ - interactive widget, never plain text
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def gen_quiz(doc_text):
    import random
    tpl = """Generate exactly 5 MCQ quiz questions from this document.
Document: {text}

EXACT format, blank line between questions:

Q: [specific question answerable from document]
A: [one possible answer]
B: [different answer]
C: [different answer]
D: [different answer]
ANSWER: [A, B, C, or D - vary which letter is correct across questions]

CRITICAL RULES:
- ANSWER must vary: use A, B, C, D across different questions - NOT all A
- Questions must be specific and answerable from document content
- All 4 options must be meaningfully different
- ANSWER line is mandatory and must match one of A/B/C/D"""
    try:
        raw = llm_call(tpl, {"text": doc_text[:3000]}, temp=0.4)
        questions = []
        for block in re.split(r"\n\s*\n", raw.strip()):
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            q, opts = {}, []
            for ln in lines:
                if ln.startswith("Q:"):
                    q["question"] = ln[2:].strip()
                elif len(ln) >= 3 and ln[1] == ":" and ln[0] in "ABCD":
                    opts.append((ln[0], ln[2:].strip()))
                elif ln.upper().startswith("ANSWER:"):
                    q["answer"] = ln.split(":", 1)[1].strip().upper()[:1]
            if q.get("question") and len(opts) >= 2:
                if "answer" not in q:
                    q["answer"] = "A"
                # Randomize option order so correct is not always first
                correct_letter = q["answer"]
                correct_text = next((o[1] for o in opts if o[0] == correct_letter), opts[0][1])
                wrong_opts = [o[1] for o in opts if o[0] != correct_letter]
                random.shuffle(wrong_opts)
                labels = ["A", "B", "C", "D"]
                correct_pos = random.randint(0, min(3, len(wrong_opts)))
                new_opts = []
                wi = 0
                for pos, lbl in enumerate(labels[:len(wrong_opts)+1]):
                    if pos == correct_pos:
                        new_opts.append((lbl, correct_text))
                    else:
                        if wi < len(wrong_opts):
                            new_opts.append((lbl, wrong_opts[wi]))
                            wi += 1
                q["options"] = new_opts
                q["answer"] = labels[correct_pos]
                questions.append(q)
        return questions[:5]
    except Exception as e:
        logger.error(f"gen_quiz: {e}")
        return []

def render_quiz(quiz, ks):
    if not quiz:
        st.warning("Could not generate quiz. Try asking again.")
        return
    st.markdown(
        "<div style='background:linear-gradient(135deg,rgba(139,92,246,0.1),rgba(99,102,241,0.06));"
        "border:1px solid rgba(139,92,246,0.25);border-radius:16px;padding:18px 22px;margin:16px 0 12px;'>"
        "<span style='font-size:0.62rem;font-weight:800;color:rgba(139,92,246,0.9);"
        "text-transform:uppercase;letter-spacing:3px;'>ðŸ“ Interactive Quiz</span>"
        "<p style='color:#64748b;font-size:0.78rem;margin:5px 0 0;'>Select your answer and click Check.</p></div>",
        unsafe_allow_html=True)
    for i, q in enumerate(quiz, 1):
        st.markdown(
            f"<div style='background:rgba(8,10,22,0.7);border:1px solid rgba(139,92,246,0.18);"
            f"border-radius:12px;padding:16px 20px;margin:10px 0;'>"
            f"<span style='color:rgba(139,92,246,0.5);font-size:0.6rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:2px;'>Question {i}</span>"
            f"<p style='color:#e2e8f0;font-size:0.9rem;font-weight:600;margin:8px 0 0;'>{q['question']}</p></div>",
            unsafe_allow_html=True)
        if q.get("options"):
            # Use hash of question text as extra key component to guarantee uniqueness
            import hashlib as _hl
            qhash = _hl.md5(q.get("question","").encode()).hexdigest()[:6]
            chosen = st.radio(f"q{i}", [f"{o[0]}.  {o[1]}" for o in q["options"]],
                              key=f"quiz_r_{i}_{ks}_{qhash}", label_visibility="collapsed", index=None)
            if st.button(f"Check Q{i}", key=f"quiz_b_{i}_{ks}_{qhash}"):
                correct = q.get("answer", "A")
                if chosen and chosen[0] == correct:
                    st.success("Correct!")
                else:
                    ct = next((o[1] for o in q["options"] if o[0] == correct), correct)
                    st.error(f"Correct: {correct}.  {ct}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RESOURCES - real clickable URLs
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def show_resources(question, answer):
    tpl = """Generate search topics for: {question}
Context: {answer}
Return ONLY this JSON (no markdown):
{{"youtube":["topic1","topic2","topic3"],"google":["phrase1","phrase2"],"scholar":["term1","term2"]}}"""
    with st.spinner("Finding resources..."):
        try:
            raw = llm_call(tpl, {"question": question, "answer": answer[:300]}, temp=0.1)
            raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
            s, e = raw.find("{"), raw.rfind("}") + 1
            k = json.loads(raw[s:e]) if s >= 0 else {}
        except Exception:
            k = {}
    if not k: st.info("No resources found."); return
    def enc(t): return t.strip().replace(" ", "+").replace("&", "%26")
    st.markdown("<div style='margin:20px 0 14px;padding:12px 16px;background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(6,182,212,0.03));border:1px solid rgba(99,102,241,0.15);border-radius:12px;'><span style='font-size:0.62rem;font-weight:800;color:rgba(99,102,241,0.8);text-transform:uppercase;letter-spacing:3px;'>Learning Resources</span></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<p style='font-size:0.73rem;font-weight:700;color:#fca5a5;margin-bottom:8px;'>YouTube</p>", unsafe_allow_html=True)
        for s in (k.get("youtube") or [])[:3]:
            st.markdown(f"<a href='https://www.youtube.com/results?search_query={enc(s)}' target='_blank' style='display:block;background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.18);border-radius:9px;padding:9px 13px;margin:5px 0;color:#fca5a5;text-decoration:none;font-size:0.78rem;font-weight:500;'>{s}</a>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p style='font-size:0.73rem;font-weight:700;color:#a5b4fc;margin-bottom:8px;'>Google</p>", unsafe_allow_html=True)
        for t in (k.get("google") or [])[:3]:
            st.markdown(f"<a href='https://www.google.com/search?q={enc(t)}' target='_blank' style='display:block;background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.15);border-radius:9px;padding:9px 13px;margin:5px 0;color:#a5b4fc;text-decoration:none;font-size:0.78rem;font-weight:500;'>{t}</a>", unsafe_allow_html=True)
    with c3:
        st.markdown("<p style='font-size:0.73rem;font-weight:700;color:#67e8f9;margin-bottom:8px;'>Scholar</p>", unsafe_allow_html=True)
        for s in (k.get("scholar") or [])[:3]:
            st.markdown(f"<a href='https://scholar.google.com/scholar?q={enc(s)}' target='_blank' style='display:block;background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.15);border-radius:9px;padding:9px 13px;margin:5px 0;color:#67e8f9;text-decoration:none;font-size:0.78rem;font-weight:500;'>{s}</a>", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FILE GENERATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def gen_file(content, fmt):
    if fmt == "txt": return content.encode("utf-8")
    if fmt == "docx":
        from docx import Document
        doc = Document(); doc.add_heading("DocuMind AI Report", 0)
        for ln in content.split("\n"):
            if ln.strip():
                if ln.startswith("## "): doc.add_heading(ln[3:], 2)
                elif ln.startswith("# "): doc.add_heading(ln[2:], 1)
                else: doc.add_paragraph(ln)
        buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=letter)
            styles = getSampleStyleSheet(); story = []
            for ln in content.split("\n"):
                if ln.strip():
                    clean = ln.replace("&","and").replace("<","").replace(">","").replace("**","").replace("*","")
                    story.append(Paragraph(clean, styles["Normal"])); story.append(Spacer(1, 4))
            doc.build(story); return buf.getvalue()
        except Exception: return content.encode("utf-8")
    if fmt == "csv":
        import pandas as pd
        rows = []
        for ln in content.split("\n"):
            if "|" in ln:
                cols = [c.strip() for c in ln.split("|") if c.strip()]
                if cols: rows.append(cols)
        if len(rows) > 1:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0]); return df.to_csv(index=False).encode("utf-8")
            except Exception: pass
        df = pd.DataFrame({"Content": [l.strip() for l in content.split("\n") if l.strip()]})
        return df.to_csv(index=False).encode("utf-8")
    if fmt == "xlsx":
        import pandas as pd
        rows = []
        for ln in content.split("\n"):
            if "|" in ln:
                cols = [c.strip() for c in ln.split("|") if c.strip()]
                if cols: rows.append(cols)
        if len(rows) > 1:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                buf = io.BytesIO(); df.to_excel(buf, index=False, engine="openpyxl"); return buf.getvalue()
            except Exception: pass
        df = pd.DataFrame({"Content": [l.strip() for l in content.split("\n") if l.strip()]})
        buf = io.BytesIO(); df.to_excel(buf, index=False, engine="openpyxl"); return buf.getvalue()
    return content.encode("utf-8")

def gen_excel_from_doc(doc_text, question):
    tpl = """Extract tabular data to answer: {question}
Document: {text}
Return ONLY valid JSON array, no markdown: [{{"Col1":"Val1","Col2":10}}]"""
    try:
        import pandas as pd
        raw = llm_call(tpl, {"text": doc_text[:4000], "question": question}, temp=0)
        raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s >= 0 and e > s:
            data = json.loads(raw[s:e])
            if data:
                df = pd.DataFrame(data); buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl"); return buf.getvalue()
    except Exception as e: logger.error(f"gen_excel: {e}")
    import pandas as pd
    df = pd.DataFrame({"Content": [l.strip() for l in doc_text.split("\n") if l.strip()][:100]})
    buf = io.BytesIO(); df.to_excel(buf, index=False, engine="openpyxl"); return buf.getvalue()

def gen_pptx(content, title="DocuMind AI"):
    try:
        from pptx import Presentation; from pptx.util import Inches
        prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = title; s.placeholders[1].text = "Generated by DocuMind AI"
        for sec in [x for x in content.split("\n\n") if x.strip()][:12]:
            lns = sec.strip().split("\n"); s2 = prs.slides.add_slide(prs.slide_layouts[1])
            s2.shapes.title.text = lns[0][:80] if lns else "Slide"
            s2.placeholders[1].text_frame.text = ("\n".join(lns[1:]))[:400] if len(lns)>1 else sec[:400]
        buf = io.BytesIO(); prs.save(buf); return buf.getvalue()
    except Exception as e: logger.error(f"gen_pptx: {e}"); return None

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TTS - st.components.v1.html (correct API)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def render_tts(text, ks=""):
    import streamlit.components.v1 as components
    clean = (text[:500].replace("\\","").replace("`","").replace('"',"")
             .replace("'","").replace("\n"," ").replace("\r","")
             .replace("<","").replace(">","").replace("&","and"))
    fp, fs = f"play_{ks}", f"stop_{ks}"
    components.html(
        f"<script>\nfunction {fp}(){{\n  window.speechSynthesis.cancel();\n"
        f"  var u=new SpeechSynthesisUtterance('{clean}');\n"
        f"  u.rate=0.92;u.pitch=1.0;u.volume=1.0;\n  window.speechSynthesis.speak(u);\n}}\n"
        f"function {fs}(){{window.speechSynthesis.cancel();}}\n</script>\n"
        f"<div style='display:flex;gap:8px;margin-top:8px;'>"
        f"<button onclick='{fp}()' style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.28);border-radius:8px;color:#a5b4fc;padding:5px 14px;font-size:0.71rem;cursor:pointer;font-weight:600;font-family:system-ui,sans-serif;'>Read Aloud</button>"
        f"<button onclick='{fs}()' style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:8px;color:#fca5a5;padding:5px 14px;font-size:0.71rem;cursor:pointer;font-weight:600;font-family:system-ui,sans-serif;'>Stop</button>"
        f"</div>",
        height=50, scrolling=False)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SMART PROMPTS - each on its own line
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def gen_prompts(text):
    tpl = """Generate exactly 6 questions a user would ask about this document.

Document: {text}

RULES:
- Each question on its OWN separate line
- Questions must be answerable from the document
- Under 9 words each
- No numbering, no bullets, no dashes
- Be specific to actual content

Questions (one per line):"""
    try:
        raw = llm_call(tpl, {"text": text[:2000]}, temp=0.4)
        lines = []
        for ln in raw.strip().split("\n"):
            ln = re.sub(r"^[\d\.\-\*\x95\u2022]\s*", "", ln.strip()).strip()
            if ln and len(ln) > 5 and not ln.startswith("#"):
                lines.append(ln)
        return lines[:6] if lines else _default_prompts()
    except Exception:
        return _default_prompts()

def _default_prompts():
    return ["What is this document about?", "Summarize the key points",
            "What are the main findings?", "What actions are recommended?",
            "What risks are mentioned?", "Who is this document for?"]

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DOCUMENT PROCESSING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _build_indexes(chunks, deps):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        fv = ex.submit(deps["create_vector_store"], chunks)
        fb = ex.submit(deps["create_bm25_index"], chunks)
        fv.result(); fb.result()

def process_doc(f, deps):
    import tempfile
    try:
        suffix = "." + f.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue()); tmp_path = tmp.name
        ok, msg = deps["validate_file"](tmp_path)
        if not ok: st.error(msg); return False
        prog = st.progress(0); status = st.empty()
        def step(p, lbl):
            prog.progress(p)
            status.markdown(f"<p style='color:#a5b4fc;font-size:0.82rem;margin:4px 0;'>âš™ {lbl}</p>", unsafe_allow_html=True)
        step(10, "Reading document...")
        text = deps["load_document"](tmp_path)
        step(30, f"Loaded {len(text):,} characters")
        step(35, "Chunking text...")
        chunks = deps["chunk_text"](text)
        step(50, f"Created {len(chunks)} chunks")
        step(55, "Building FAISS + BM25 in parallel...")
        _build_indexes(chunks, deps)
        step(85, "Indexes ready")
        step(90, "Generating smart prompts...")
        prompts = gen_prompts(text)
        step(100, "Done!")
        prog.empty(); status.empty()
        st.session_state.file_path = tmp_path; st.session_state.doc_name = f.name
        st.session_state.doc_text = text; st.session_state.prompts = prompts
        st.session_state.doc_ready = True; st.session_state.chat = []
        st.session_state.pending = {}; st.session_state.quiz_store = {}
        st.session_state.memory = st.session_state.fmm.get_memory(f.name)
        existing = [d["name"] for d in st.session_state.multi_docs]
        if f.name not in existing:
            st.session_state.multi_docs.append({"name": f.name, "path": tmp_path, "text": text, "chunks": chunks})
        else:
            for d in st.session_state.multi_docs:
                if d["name"] == f.name: d.update({"path": tmp_path, "text": text, "chunks": chunks})
        return True
    except Exception as e:
        logger.error(f"process_doc: {e}"); st.error(str(e)); return False

def process_multi_parallel(files, deps):
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed
    prog = st.progress(0); status = st.empty(); total = len(files); done = 0; all_docs = []
    def proc_one(f):
        suffix = "." + f.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue()); p = tmp.name
        ok, msg = deps["validate_file"](p)
        if not ok: return f.name, None, msg
        text = deps["load_document"](p); chunks = deps["chunk_text"](text)
        return f.name, {"name": f.name, "path": p, "text": text, "chunks": chunks}, None
    with ThreadPoolExecutor(max_workers=min(total, 3)) as ex:
        futs = {ex.submit(proc_one, f): f.name for f in files}
        for fut in as_completed(futs):
            nm, doc, err = fut.result(); done += 1
            prog.progress(int(done / total * 65))
            if err: status.markdown(f"<p style='color:#fca5a5;font-size:0.82rem;'>âš  {nm}: {err}</p>", unsafe_allow_html=True)
            else:
                status.markdown(f"<p style='color:#a5b4fc;font-size:0.82rem;'>âœ“ {nm}</p>", unsafe_allow_html=True)
                all_docs.append(doc)
    if all_docs:
        prog.progress(70); status.markdown("<p style='color:#a5b4fc;font-size:0.82rem;'>âš™ Building indexes...</p>", unsafe_allow_html=True)
        active = all_docs[-1]; _build_indexes(active["chunks"], deps)
        existing = [d["name"] for d in st.session_state.multi_docs]
        for doc in all_docs:
            if doc["name"] not in existing: st.session_state.multi_docs.append(doc)
        prog.progress(90); status.markdown("<p style='color:#a5b4fc;font-size:0.82rem;'>âš™ Generating prompts...</p>", unsafe_allow_html=True)
        st.session_state.file_path = active["path"]; st.session_state.doc_name = active["name"]
        st.session_state.doc_text = active["text"]; st.session_state.prompts = gen_prompts(active["text"])
        st.session_state.doc_ready = True; st.session_state.chat = []
        st.session_state.pending = {}; st.session_state.quiz_store = {}
        st.session_state.memory = st.session_state.fmm.get_memory(active["name"])
    prog.progress(100); time.sleep(0.3); prog.empty(); status.empty()
    return len(all_docs)

def switch_document(doc, deps):
    """Rebuild indexes when switching so answers use the correct document."""
    with st.spinner(f"Switching to {doc['name']}..."):
        chunks = doc.get("chunks")
        if not chunks:
            chunks = deps["chunk_text"](doc["text"]); doc["chunks"] = chunks
        _build_indexes(chunks, deps)
    st.session_state.doc_name = doc["name"]; st.session_state.file_path = doc["path"]
    st.session_state.doc_text = doc["text"]; st.session_state.pending = {}
    st.session_state.quiz_store = {}
    st.session_state.memory = st.session_state.fmm.get_memory(doc["name"])

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ANSWER ENGINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def answer_q(question, deps):
    ok, msg = deps["validate_question"](question)
    if not ok: st.warning(msg); return
    if st.session_state.get("_stop_flag", False):
        st.session_state["_stop_flag"] = False
        return
    intent = detect_intent(question)
    st.session_state["_processing"] = True
    st.session_state.chat.append({"role": "human", "content": question})
    if intent["quiz"]:
        with st.spinner("Generating quiz questions..."):
            quiz = gen_quiz(st.session_state.doc_text)
        ks = uid(); st.session_state.quiz_store[ks] = quiz
        marker = f"__QUIZ__{ks}"
        st.session_state.chat.append({"role": "assistant", "content": marker})
        st.session_state.pending = {"evidence": [], "intent": intent, "question": question, "answer": marker, "ks": ks}
        return
    answer = ""; evidence = []
    with st.spinner("Analysing..."):
        try:
            result = deps["run_workflow"](question=question, memory=st.session_state.memory,
                                          file_path=st.session_state.file_path, answer_mode=st.session_state.answer_mode)
            if isinstance(result, dict):
                answer = result.get("answer","") or result.get("output","") or ""; evidence = result.get("evidence",[]) or []
            elif isinstance(result, str): answer = result
            else: answer = str(result) if result else ""
            answer = answer.strip()
            if not answer or answer.lower() in {"none","result: none","none.","","n/a"} or answer.startswith("Result: None"):
                answer = direct_summarize(st.session_state.doc_text, question, st.session_state.answer_mode)
        except Exception as e:
            logger.error(f"workflow: {e}")
            answer = direct_summarize(st.session_state.doc_text, question, st.session_state.answer_mode)
    if not answer or not answer.strip():
        answer = direct_summarize(st.session_state.doc_text, question, st.session_state.answer_mode)
    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.last_q = question; st.session_state.last_a = answer
    st.session_state["_processing"] = False
    st.session_state.pending = {"evidence": evidence, "intent": intent, "question": question, "answer": answer, "ks": uid()}

def run_quick_action(action, deps):
    st.session_state.chat.append({"role": "human", "content": action})
    if "quiz" in action.lower():
        with st.spinner("Generating quiz..."):
            quiz = gen_quiz(st.session_state.doc_text)
        ks = uid(); st.session_state.quiz_store[ks] = quiz; marker = f"__QUIZ__{ks}"
        st.session_state.chat.append({"role": "assistant", "content": marker})
        st.session_state.pending = {"evidence": [], "intent": {"quiz": True}, "question": action, "answer": marker, "ks": ks}
        return
    with st.spinner("Working on it..."):
        result = direct_summarize(st.session_state.doc_text, action, st.session_state.answer_mode)
    if not result or not result.strip(): result = "Could not generate response. Please try again."
    st.session_state.chat.append({"role": "assistant", "content": result})
    st.session_state.last_q = action; st.session_state.last_a = result
    st.session_state.pending = {"evidence": [], "intent": {}, "question": action, "answer": result, "ks": uid()}

def render_pending(p):
    if not p: return
    ev = p.get("evidence", []); intent = p.get("intent", {}); question = p.get("question", "")
    answer = p.get("answer", ""); ks = p.get("ks", uid())
    if ev:
        with st.expander("Evidence Sources", expanded=False):
            for i, chunk in enumerate(ev, 1):
                st.markdown(f"<div style='background:rgba(99,102,241,0.04);border-left:2px solid rgba(99,102,241,0.3);border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;font-size:0.79rem;color:#64748b;'><strong style='color:#a5b4fc;'>Source {i}</strong><br>{chunk}</div>", unsafe_allow_html=True)
    # Quiz is rendered in the chat history loop (not here) to avoid duplicate keys
    if intent.get("quiz"):
        return  # quiz already rendered in chat loop above
    if intent.get("chart"):
        with st.spinner("Generating charts..."):
            charts = gen_charts(st.session_state.doc_text, question)
        if charts:
            if len(charts) == 1: st.plotly_chart(charts[0][1], use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                for i, (t, f) in enumerate(charts[:4]):
                    with (c1 if i % 2 == 0 else c2): st.plotly_chart(f, use_container_width=True)
        else: st.info("No numerical data found. Try: 'show a bar chart of [column name]'.")
    if intent.get("excel"):
        st.markdown("<div style='margin:14px 0 6px;'><span style='font-size:0.62rem;font-weight:700;color:rgba(16,185,129,0.8);text-transform:uppercase;letter-spacing:2px;'>Excel Export</span></div>", unsafe_allow_html=True)
        with st.spinner("Creating Excel..."): xb = gen_excel_from_doc(st.session_state.doc_text, question)
        st.download_button("Download Excel", data=xb, file_name="documind_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dlx_{ks}")
    elif intent.get("pptx"):
        with st.spinner("Creating PowerPoint..."): pb = gen_pptx(answer, st.session_state.doc_name)
        if pb: st.download_button("Download PowerPoint", data=pb, file_name="documind.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", key=f"dlp_{ks}")
    elif intent.get("export_doc"):
        st.markdown("<div style='margin:18px 0 10px;padding:12px 16px;background:linear-gradient(135deg,rgba(16,185,129,0.06),rgba(6,182,212,0.03));border:1px solid rgba(16,185,129,0.15);border-radius:12px;'><span style='font-size:0.62rem;font-weight:800;color:rgba(16,185,129,0.9);text-transform:uppercase;letter-spacing:3px;'>Download Answer</span></div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, (fmt, lbl, mime) in enumerate([("txt","TXT","text/plain"),("docx","DOCX","application/vnd.openxmlformats-officedocument.wordprocessingml.document"),("pdf","PDF","application/pdf"),("csv","CSV","text/csv"),("xlsx","Excel","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")]):
            with cols[i]: st.download_button(lbl, data=gen_file(answer, fmt), file_name=f"documind.{fmt}", mime=mime, key=f"dl_{fmt}_{ks}")
    if intent.get("regen_doc"):
        st.markdown("<div style='margin:18px 0 10px;padding:12px 16px;background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(139,92,246,0.04));border:1px solid rgba(99,102,241,0.2);border-radius:12px;'><span style='font-size:0.62rem;font-weight:800;color:rgba(99,102,241,0.9);text-transform:uppercase;letter-spacing:3px;'>Regenerated Document</span></div>", unsafe_allow_html=True)
        with st.spinner("Generating document files..."): files_dict, _ = regen_document(st.session_state.doc_text, question)
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1: st.download_button("DOCX", data=files_dict["docx"], file_name="documind_regen.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"rg_docx_{ks}")
        with rc2: st.download_button("PDF",  data=files_dict["pdf"],  file_name="documind_regen.pdf",  mime="application/pdf", key=f"rg_pdf_{ks}")
        with rc3: st.download_button("CSV",  data=files_dict["csv"],  file_name="documind_regen.csv",  mime="text/csv", key=f"rg_csv_{ks}")
        with rc4: st.download_button("TXT",  data=files_dict["txt"],  file_name="documind_regen.txt",  mime="text/plain", key=f"rg_txt_{ks}")
    if intent.get("resources"): show_resources(question, answer)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CSS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
CSS = """<style>
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',Arial,sans-serif !important;}
.stApp{background:radial-gradient(ellipse at 20% 20%,rgba(99,102,241,0.05) 0%,transparent 55%),radial-gradient(ellipse at 80% 80%,rgba(6,182,212,0.04) 0%,transparent 55%),#04060f !important;}
section[data-testid="stSidebar"]{background:rgba(4,6,15,0.98) !important;border-right:1px solid rgba(99,102,241,0.1) !important;}
section[data-testid="stSidebar"] *{color:#94a3b8 !important;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#6366f1) !important;color:#fff !important;border:1px solid rgba(99,102,241,0.4) !important;border-radius:10px !important;font-weight:600 !important;font-size:0.79rem !important;transition:all 0.2s !important;box-shadow:0 2px 10px rgba(79,70,229,0.2) !important;}
.stButton>button:hover{background:linear-gradient(135deg,#6366f1,#818cf8) !important;transform:translateY(-2px) !important;box-shadow:0 5px 18px rgba(99,102,241,0.35) !important;}
.stChatInputContainer{background:rgba(8,10,22,0.9) !important;border:1px solid rgba(99,102,241,0.2) !important;border-radius:16px !important;box-shadow:0 4px 20px rgba(0,0,0,0.25) !important;}
.stChatInputContainer:focus-within{border-color:rgba(99,102,241,0.45) !important;box-shadow:0 0 0 3px rgba(99,102,241,0.09) !important;}
.stChatInputContainer textarea{background:transparent !important;color:#e2e8f0 !important;}
div[data-testid="stChatMessage"]{background:transparent !important;border-radius:14px !important;}
div[data-testid="stExpander"]{background:rgba(8,10,22,0.6) !important;border:1px solid rgba(99,102,241,0.1) !important;border-radius:12px !important;}
div[data-testid="stSelectbox"]>div>div{background:rgba(8,10,22,0.9) !important;border:1px solid rgba(99,102,241,0.18) !important;border-radius:10px !important;}
section[data-testid="stFileUploaderDropzone"]{background:rgba(99,102,241,0.02) !important;border:2px dashed rgba(99,102,241,0.25) !important;border-radius:14px !important;}
div[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#4f46e5,#7c3aed,#06b6d4) !important;border-radius:6px !important;box-shadow:0 0 10px rgba(99,102,241,0.35) !important;}
div[data-testid="stSuccess"]{background:rgba(16,185,129,0.07) !important;border:1px solid rgba(16,185,129,0.22) !important;border-radius:11px !important;}
div[data-testid="stError"]{background:rgba(239,68,68,0.07) !important;border:1px solid rgba(239,68,68,0.22) !important;border-radius:11px !important;}
div[data-testid="stInfo"]{background:rgba(99,102,241,0.06) !important;border:1px solid rgba(99,102,241,0.18) !important;border-radius:11px !important;}
div[data-testid="stWarning"]{background:rgba(245,158,11,0.06) !important;border:1px solid rgba(245,158,11,0.2) !important;border-radius:11px !important;}
div[data-testid="stDownloadButton"]>button{background:rgba(16,185,129,0.09) !important;border:1px solid rgba(16,185,129,0.25) !important;color:#6ee7b7 !important;border-radius:10px !important;font-weight:600 !important;}
div[data-testid="stDownloadButton"]>button:hover{background:rgba(16,185,129,0.16) !important;transform:translateY(-1px) !important;}
div[data-testid="stRadio"] label{background:rgba(8,10,22,0.55) !important;border:1px solid rgba(255,255,255,0.06) !important;border-radius:9px !important;padding:8px 14px !important;margin:3px 0 !important;color:#94a3b8 !important;}
::-webkit-scrollbar{width:3px;height:3px;}::-webkit-scrollbar-track{background:transparent;}::-webkit-scrollbar-thumb{background:rgba(99,102,241,0.22);border-radius:4px;}
hr{border-color:rgba(255,255,255,0.04) !important;margin:10px 0 !important;}
</style>"""

SIDEBAR_CSS = """<style>
.dm-logo{display:flex;align-items:center;gap:12px;padding:18px 0 16px;border-bottom:1px solid rgba(99,102,241,0.1);margin-bottom:18px;}
/* .dm-icon replaced by SVG logo */
.dm-name{font-size:1rem !important;font-weight:800 !important;color:#f1f5f9 !important;line-height:1.2;}
.dm-sub{font-size:0.56rem !important;color:rgba(99,102,241,0.7) !important;font-weight:700 !important;text-transform:uppercase;letter-spacing:2.5px;}
.sec-lbl{font-size:0.56rem !important;font-weight:800 !important;color:rgba(99,102,241,0.5) !important;text-transform:uppercase !important;letter-spacing:3px !important;display:block !important;margin:14px 0 8px !important;}
.doc-card{background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(6,182,212,0.03));border:1px solid rgba(99,102,241,0.13);border-radius:11px;padding:9px 13px;margin:5px 0;}
.doc-name{color:#a5b4fc !important;font-size:0.78rem !important;font-weight:600 !important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:195px;}
.doc-size{color:#3f4f6b !important;font-size:0.66rem !important;margin-top:2px;}
.msg-c{background:linear-gradient(135deg,rgba(99,102,241,0.07),rgba(6,182,212,0.04));border:1px solid rgba(99,102,241,0.11);border-radius:11px;padding:13px;text-align:center;margin:6px 0;}
.mc-n{font-size:2rem !important;font-weight:900 !important;background:linear-gradient(135deg,#a5b4fc,#06b6d4) !important;-webkit-background-clip:text !important;-webkit-text-fill-color:transparent !important;background-clip:text !important;display:block !important;line-height:1 !important;}
.mc-l{font-size:0.56rem !important;color:#3f4f6b !important;text-transform:uppercase !important;letter-spacing:2.5px !important;margin-top:4px !important;display:block !important;}
</style>"""

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HERO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@st.cache_data(show_spinner=False)
def _hero_html():
    return """<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
*{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;}
body{background:radial-gradient(ellipse at 30% 20%,rgba(99,102,241,0.07) 0%,transparent 55%),radial-gradient(ellipse at 70% 80%,rgba(6,182,212,0.05) 0%,transparent 55%),#04060f;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:48px 24px;text-align:center;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.15}}
@keyframes flow{0%{background-position:0% center}100%{background-position:300% center}}
@keyframes up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);border-radius:100px;padding:7px 20px;font-size:0.61rem;font-weight:700;color:#818cf8;letter-spacing:3px;text-transform:uppercase;margin-bottom:30px;animation:up 0.4s ease forwards;}
.dot{width:5px;height:5px;background:#22c55e;border-radius:50%;animation:blink 2s ease infinite;box-shadow:0 0 6px #22c55e;}
h1{font-size:clamp(2.6rem,7vw,5.4rem);font-weight:900;letter-spacing:-3.5px;line-height:0.96;margin-bottom:20px;background:linear-gradient(135deg,#fff 0%,#c7d2fe 20%,#a5b4fc 40%,#06b6d4 65%,#a5b4fc 80%,#fff 100%);background-size:300% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:flow 5s linear infinite,up 0.6s ease forwards;}
.sub{font-size:0.98rem;color:#374151;max-width:440px;line-height:1.85;margin:0 auto 44px;animation:up 0.8s ease forwards;}
.sub b{color:#6366f1;font-weight:600;}
.stats{display:flex;border:1px solid rgba(255,255,255,0.06);border-radius:14px;overflow:hidden;background:rgba(8,10,22,0.85);max-width:520px;margin:0 auto 52px;animation:up 1s ease forwards;}
.s{flex:1;padding:18px 8px;text-align:center;border-right:1px solid rgba(255,255,255,0.05);}
.s:last-child{border-right:none;}
.sv{font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#a5b4fc,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;display:block;}
.sl{font-size:0.55rem;color:#1e293b;text-transform:uppercase;letter-spacing:2px;margin-top:3px;display:block;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;max-width:880px;margin:0 auto;animation:up 1.2s ease forwards;}
.card{background:rgba(8,10,22,0.7);border:1px solid rgba(255,255,255,0.05);border-radius:13px;padding:18px 14px;text-align:left;transition:all 0.28s;}
.card:hover{border-color:rgba(99,102,241,0.28);transform:translateY(-4px);box-shadow:0 14px 35px rgba(99,102,241,0.13);}
.num{font-size:0.57rem;font-weight:800;color:rgba(99,102,241,0.38);letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;}
.ico{font-size:1rem;margin-bottom:7px;display:block;}
.nm{color:#e2e8f0;font-size:0.82rem;font-weight:700;margin-bottom:4px;}
.tx{color:#1e293b;font-size:0.7rem;line-height:1.55;}
</style></head><body>
<div class="badge"><div class="dot"></div>Enterprise Document Intelligence</div>
<h1>DocuMind AI</h1>
<p class="sub">Upload any document. Ask in <b>plain English</b>.<br>Get precise answers, charts, quizzes, and exports instantly.</p>
<div class="stats">
<div class="s"><span class="sv">6+</span><span class="sl">Formats</span></div>
<div class="s"><span class="sv">5</span><span class="sl">AI Agents</span></div>
<div class="s"><span class="sv">20+</span><span class="sl">Features</span></div>
<div class="s"><span class="sv">100%</span><span class="sl">Private</span></div></div>
<div class="grid">
<div class="card"><div class="num">01</div><span class="ico">ðŸ“Š</span><div class="nm">Auto Charts</div><div class="tx">Ask "show chart" for interactive Plotly graphs</div></div>
<div class="card"><div class="num">02</div><span class="ico">ðŸ“</span><div class="nm">Quiz Mode</div><div class="tx">Ask "give me a quiz" for MCQ with instant checking</div></div>
<div class="card"><div class="num">03</div><span class="ico">ðŸ”Š</span><div class="nm">Read Aloud</div><div class="tx">Browser TTS reads every answer aloud</div></div>
<div class="card"><div class="num">04</div><span class="ico">ðŸ”—</span><div class="nm">Live Resources</div><div class="tx">Ask "show resources" for YouTube, Google, Scholar</div></div>
<div class="card"><div class="num">05</div><span class="ico">â¬‡</span><div class="nm">Export Any Format</div><div class="tx">Ask "export as pdf/docx/excel" to download</div></div>
<div class="card"><div class="num">06</div><span class="ico">ðŸ”„</span><div class="nm">Regenerate Docs</div><div class="tx">Ask "shorten to 10 records and regenerate"</div></div>
<div class="card"><div class="num">07</div><span class="ico">ðŸ”</span><div class="nm">Hybrid Search</div><div class="tx">FAISS semantic + BM25 keyword fusion</div></div>
<div class="card"><div class="num">08</div><span class="ico">ðŸ—‚</span><div class="nm">Multi-Document</div><div class="tx">Upload multiple files, switch, compare all</div></div>
</div></body></html>"""

def show_hero():
    import streamlit.components.v1 as components
    components.html(_hero_html(), height=880, scrolling=False)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MULTI-DOC HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def query_multi_direct(question, docs):
    combined = ""
    for i, d in enumerate(docs[:5], 1):
        combined += f"\n\n{'='*50}\nDOCUMENT {i}: {d['name']}\n{'='*50}\n{d['text'][:2500]}"
    tpl = """You are analyzing {n} documents simultaneously.

{docs}

Question: {question}

Instructions:
- Answer from ALL documents - don't skip any
- For each relevant fact, clearly state: "(From: Document Name)"
- Compare information across documents when relevant
- If documents have different info on same topic, highlight the differences
- Give a comprehensive unified answer
- End with: Summary: [one paragraph synthesizing all documents]

Answer:"""
    try:
        return llm_call(tpl, {"docs": combined, "question": question, "n": str(len(docs))}, temp=0.3)
    except Exception as e:
        return f"Error querying documents: {e}"

def compare_docs_direct(docs):
    combined = ""
    for i, d in enumerate(docs[:5], 1):
        combined += f"\n\n{'='*50}\nDOCUMENT {i}: {d['name']}\n{'='*50}\n{d['text'][:2000]}"
    tpl = """Compare these {n} documents in detail.

{docs}

Create a comprehensive comparison covering:
1. **Purpose & Type** of each document
2. **Key Similarities** across documents
3. **Key Differences** between documents  
4. **Unique Information** in each document
5. **Recommendations** - which document is best for what purpose

Use markdown headers and tables for clarity.
End with: **Overall Summary:** [2-3 sentences]"""
    try:
        return llm_call(tpl, {"docs": combined, "n": str(len(docs))}, temp=0.3)
    except Exception as e:
        return f"Error comparing documents: {e}"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _show_upload_sidebar_only():
    """Lightweight sidebar shown on hero page â€” zero heavy imports."""
    with st.sidebar:
        st.markdown("""
<div class='dm-logo'>
  <div style='flex-shrink:0;'>
    <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="bgGrad" x1="0" y1="0" x2="46" y2="46" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#0d0b1e"/>
          <stop offset="30%" stop-color="#2d2a8a"/>
          <stop offset="65%" stop-color="#5b21b6"/>
          <stop offset="100%" stop-color="#0e7490"/>
        </linearGradient>
        <linearGradient id="shineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="white" stop-opacity="0.13"/>
          <stop offset="100%" stop-color="white" stop-opacity="0"/>
        </linearGradient>
        <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#22d3ee"/>
          <stop offset="100%" stop-color="#0891b2"/>
        </radialGradient>
        <filter id="nodeGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur"/>
          <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
        <filter id="softGlow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.2" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <clipPath id="rounded"><rect width="46" height="46" rx="12"/></clipPath>
      </defs>
      <rect x="1" y="1" width="44" height="44" rx="12" fill="#4f46e5" opacity="0.35" filter="url(#nodeGlow)"/>
      <rect width="46" height="46" rx="12" fill="url(#bgGrad)"/>
      <rect width="46" height="26" rx="12" fill="url(#shineGrad)" clip-path="url(#rounded)"/>
      <rect x="0.75" y="0.75" width="44.5" height="44.5" rx="11.5" fill="none" stroke="white" stroke-opacity="0.1" stroke-width="1.5"/>
      <rect x="10" y="11" width="3.5" height="24" rx="1.75" fill="white" opacity="0.92"/>
      <path d="M 13.5 11.5 C 26 11.5 32.5 16.5 32.5 23 C 32.5 29.5 26 34.5 13.5 34.5" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" opacity="0.88"/>
      <rect x="17" y="17.5" width="8" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.9"/>
      <rect x="17" y="22" width="11" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.72"/>
      <rect x="17" y="26.5" width="7" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.55"/>
      <circle cx="37" cy="12" r="5.5" fill="#06b6d4" opacity="0.12" filter="url(#nodeGlow)"/>
      <circle cx="37" cy="12" r="4" fill="none" stroke="#22d3ee" stroke-width="1.2" stroke-opacity="0.6"/>
      <circle cx="37" cy="12" r="2.6" fill="url(#nodeGrad)" filter="url(#softGlow)" opacity="0.98"/>
      <circle cx="37" cy="12" r="1.2" fill="white" opacity="0.9"/>
      <line x1="31" y1="15" x2="34" y2="13.5" stroke="#22d3ee" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="1.5 1.5"/>
      <rect x="10" y="37.5" width="26" height="1.2" rx="0.6" fill="url(#nodeGrad)" opacity="0.35"/>
    </svg>
  </div>
  <div><div class='dm-name'>DocuMind AI</div><div class='dm-sub'>Document Intelligence</div></div>
</div>""", unsafe_allow_html=True)
        st.markdown("<span class='sec-lbl'>Upload Document(s)</span>", unsafe_allow_html=True)
        files = st.file_uploader(
            "upload", type=["pdf","docx","pptx","ppt","xlsx","csv","txt"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if files:
            for f in files:
                st.markdown(
                    f"<div class='doc-card'><div class='doc-name'>{f.name}</div>"
                    f"<div class='doc-size'>{f.size/1024:.0f} KB</div></div>",
                    unsafe_allow_html=True)
            lbl = f"Process {len(files)} Files" if len(files) > 1 else "Process Document"
            if st.button(lbl, use_container_width=True):
                # Signal that process was clicked â€” next rerun will load deps
                st.session_state["_process_clicked"] = True
                st.session_state["_pending_files"] = files
                st.rerun()


def main():
    # Inject CSS instantly â€” pure strings, zero imports
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    # â”€â”€ FAST PATH: show hero immediately without any heavy imports â”€â”€
    # load_core() imports sentence-transformers, FAISS, LangGraph, pdfplumber
    # which adds 3-8 seconds on cold start. We defer it until needed.
    _doc_ready = st.session_state.get("doc_ready", False)
    _process_clicked = st.session_state.get("_process_clicked", False)

    if not _doc_ready and not _process_clicked:
        # Show sidebar upload UI + hero page â€” no heavy imports at all
        _show_upload_sidebar_only()
        show_hero()
        return

    # Only reach here when user has a document or clicked Process
    deps = load_core()
    init(deps)

    # Handle deferred file processing (clicked Process on hero page)
    if st.session_state.get("_process_clicked") and not st.session_state.get("doc_ready"):
        pending = st.session_state.get("_pending_files", [])
        st.session_state["_process_clicked"] = False
        st.session_state["_pending_files"] = []
        if pending:
            if len(pending) == 1:
                if process_doc(pending[0], deps):
                    st.success(f"Ready: {pending[0].name}")
                    st.rerun()
            else:
                n = process_multi_parallel(pending, deps)
                if n > 0:
                    st.success(f"{n} documents ready")
                    st.rerun()

    with st.sidebar:
        st.markdown("""
<div class='dm-logo'>
  <div style='flex-shrink:0;'>
    <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="bgGrad" x1="0" y1="0" x2="46" y2="46" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#0d0b1e"/>
          <stop offset="30%" stop-color="#2d2a8a"/>
          <stop offset="65%" stop-color="#5b21b6"/>
          <stop offset="100%" stop-color="#0e7490"/>
        </linearGradient>
        <linearGradient id="shineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="white" stop-opacity="0.13"/>
          <stop offset="100%" stop-color="white" stop-opacity="0"/>
        </linearGradient>
        <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#22d3ee"/>
          <stop offset="100%" stop-color="#0891b2"/>
        </radialGradient>
        <filter id="nodeGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur"/>
          <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
        <filter id="softGlow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.2" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <clipPath id="rounded"><rect width="46" height="46" rx="12"/></clipPath>
      </defs>
      <rect x="1" y="1" width="44" height="44" rx="12" fill="#4f46e5" opacity="0.35" filter="url(#nodeGlow)"/>
      <rect width="46" height="46" rx="12" fill="url(#bgGrad)"/>
      <rect width="46" height="26" rx="12" fill="url(#shineGrad)" clip-path="url(#rounded)"/>
      <rect x="0.75" y="0.75" width="44.5" height="44.5" rx="11.5" fill="none" stroke="white" stroke-opacity="0.1" stroke-width="1.5"/>
      <rect x="10" y="11" width="3.5" height="24" rx="1.75" fill="white" opacity="0.92"/>
      <path d="M 13.5 11.5 C 26 11.5 32.5 16.5 32.5 23 C 32.5 29.5 26 34.5 13.5 34.5" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" opacity="0.88"/>
      <rect x="17" y="17.5" width="8" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.9"/>
      <rect x="17" y="22" width="11" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.72"/>
      <rect x="17" y="26.5" width="7" height="1.8" rx="0.9" fill="url(#nodeGrad)" opacity="0.55"/>
      <circle cx="37" cy="12" r="5.5" fill="#06b6d4" opacity="0.12" filter="url(#nodeGlow)"/>
      <circle cx="37" cy="12" r="4" fill="none" stroke="#22d3ee" stroke-width="1.2" stroke-opacity="0.6"/>
      <circle cx="37" cy="12" r="2.6" fill="url(#nodeGrad)" filter="url(#softGlow)" opacity="0.98"/>
      <circle cx="37" cy="12" r="1.2" fill="white" opacity="0.9"/>
      <line x1="31" y1="15" x2="34" y2="13.5" stroke="#22d3ee" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="1.5 1.5"/>
      <rect x="10" y="37.5" width="26" height="1.2" rx="0.6" fill="url(#nodeGrad)" opacity="0.35"/>
    </svg>
  </div>
  <div><div class='dm-name'>DocuMind AI</div><div class='dm-sub'>Document Intelligence</div></div>
</div>""", unsafe_allow_html=True)
        st.markdown("<span class='sec-lbl'>Upload Document(s)</span>", unsafe_allow_html=True)
        files = st.file_uploader("upload", type=["pdf","docx","pptx","ppt","xlsx","csv","txt"],
                                  accept_multiple_files=True, label_visibility="collapsed")
        if files:
            for f in files:
                st.markdown(f"<div class='doc-card'><div class='doc-name'>{f.name}</div><div class='doc-size'>{f.size/1024:.0f} KB</div></div>", unsafe_allow_html=True)
            lbl = f"Process {len(files)} Files" if len(files) > 1 else "Process Document"
            if st.button(lbl, use_container_width=True):
                if len(files) == 1:
                    if process_doc(files[0], deps): st.success(f"Ready: {files[0].name}"); st.rerun()
                else:
                    n = process_multi_parallel(files, deps)
                    if n > 0: st.success(f"{n} documents ready"); st.rerun()

        if st.session_state.doc_ready:
            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Answer Style</span>", unsafe_allow_html=True)
            mode = st.selectbox("mode", options=["detailed","quick","bullet","beginner","executive","table"],
                format_func=lambda x: {"detailed":"ðŸ” Detailed","quick":"âš¡ Quick","bullet":"â€¢ Bullets","beginner":"ðŸ‘¶ Beginner","executive":"ðŸŽ¯ Executive","table":"ðŸ“‹ Table"}[x],
                label_visibility="collapsed")
            st.session_state.answer_mode = mode
            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Active Document</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='doc-card'><div class='doc-name'>ðŸ“„ {st.session_state.doc_name}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='msg-c'><span class='mc-n'>{len(st.session_state.chat)}</span><span class='mc-l'>Messages</span></div>", unsafe_allow_html=True)
            if len(st.session_state.multi_docs) > 1:
                st.markdown("---")
                st.markdown("<span class='sec-lbl'>All Documents</span>", unsafe_allow_html=True)
                for i, doc in enumerate(st.session_state.multi_docs):
                    is_active = doc["name"] == st.session_state.doc_name
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        color = "#3fb950" if is_active else "#64748b"
                        marker = "â—" if is_active else "â—‹"
                        st.markdown(f"<div style='color:{color};font-size:0.77rem;padding:3px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{marker} {doc['name']}</div>", unsafe_allow_html=True)
                    with c2:
                        if not is_active and st.button("Use", key=f"sw_{i}"):
                            switch_document(doc, deps); st.rerun()
            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Quick Actions</span>", unsafe_allow_html=True)
            for i, act in enumerate(["Summarize document","Extract action items","Identify all risks","Extract key metrics","Generate FAQ","Create a quiz"]):
                if st.button(act, use_container_width=True, key=f"qa_{i}"):
                    run_quick_action(act, deps); st.rerun()
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Clear", use_container_width=True):
                    st.session_state.memory.clear(); st.session_state.chat = []
                    st.session_state.pending = {}; st.session_state.quiz_store = {}; st.rerun()
            with cc2:
                if st.button("New File", use_container_width=True):
                    for k in ["doc_ready","file_path","doc_name","doc_text","chat","prompts","multi_docs","pending","quiz_store"]:
                        st.session_state[k] = ([] if k in ["chat","prompts","multi_docs"] else {} if k in ["pending","quiz_store"] else False if k == "doc_ready" else "")
                    st.session_state.memory = deps["SessionMemory"](); st.rerun()

    if not st.session_state.doc_ready:
        show_hero(); return

    if st.session_state.prompts:
        st.markdown("<p style='color:rgba(99,102,241,0.55);font-size:0.61rem;font-weight:800;text-transform:uppercase;letter-spacing:3px;margin-bottom:10px;'>Suggested</p>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, prompt in enumerate(st.session_state.prompts):
            with cols[i % 3]:
                if st.button(prompt, key=f"sp_{i}", use_container_width=True):
                    answer_q(prompt, deps); st.rerun()
        st.markdown("---")

    if len(st.session_state.multi_docs) > 1:
        with st.expander(f"Multi-Document Panel - {len(st.session_state.multi_docs)} loaded", expanded=False):
            badge_html = "".join([f"<span style='background:rgba({'99,102,241' if d['name']==st.session_state.doc_name else '30,41,59'},0.4);border:1px solid rgba({'99,102,241' if d['name']==st.session_state.doc_name else '51,65,85'},0.3);border-radius:7px;padding:4px 10px;font-size:0.72rem;color:{'#a5b4fc' if d['name']==st.session_state.doc_name else '#64748b'};margin:3px;display:inline-block;'>{'âœ…' if d['name']==st.session_state.doc_name else 'ðŸ“„'} {d['name']}</span>" for d in st.session_state.multi_docs])
            st.markdown(f"<div style='margin-bottom:12px;'>{badge_html}</div>", unsafe_allow_html=True)
            mq = st.text_input("Ask across all documents:", key="mq_input")
            mc1, mc2 = st.columns(2)
            with mc1:
                if st.button("Query All", use_container_width=True, key="btn_qall"):
                    if mq:
                        with st.spinner(f"Querying {len(st.session_state.multi_docs)} documents..."):
                            ans = query_multi_direct(mq, st.session_state.multi_docs)
                        st.session_state.chat.append({"role":"human","content":f"[All Docs] {mq}"})
                        st.session_state.chat.append({"role":"assistant","content":ans})
                        st.session_state.pending = {"evidence":[],"intent":{},"question":mq,"answer":ans,"ks":uid()}; st.rerun()
                    else:
                        st.warning("Type a question above to query all documents.")
            with mc2:
                if st.button("Compare All", use_container_width=True, key="btn_cmpall"):
                    with st.spinner(f"Comparing {len(st.session_state.multi_docs)} documents..."):
                        comp = compare_docs_direct(st.session_state.multi_docs)
                    st.session_state.chat.append({"role":"human","content":"Compare all uploaded documents"})
                    st.session_state.chat.append({"role":"assistant","content":comp})
                    st.session_state.pending = {"evidence":[],"intent":{},"question":"compare","answer":comp,"ks":uid()}; st.rerun()

    for idx, msg in enumerate(st.session_state.chat):
        if msg["role"] == "human":
            with st.chat_message("user"): st.markdown(msg["content"])
        else:
            content = msg["content"]
            if content.startswith("__QUIZ__"):
                ks = content.replace("__QUIZ__", "")
                with st.chat_message("assistant"):
                    if ks in st.session_state.quiz_store: render_quiz(st.session_state.quiz_store[ks], ks)
                    else: st.markdown("Quiz expired. Ask again to regenerate.")
            else:
                with st.chat_message("assistant"):
                    st.markdown(
                        f"<div style='max-width:860px;line-height:1.75;"
                        f"font-size:0.92rem;color:#e2e8f0;'>{content}</div>",
                        unsafe_allow_html=True
                    )
                    if idx == len(st.session_state.chat) - 1:
                        render_tts(content, ks=str(idx))

    render_pending(st.session_state.get("pending", {}))

    # Stop/Cancel button row
    if st.session_state.get("_processing", False):
        if st.button("Stop Processing", type="secondary", use_container_width=True):
            st.session_state["_processing"] = False
            st.session_state["_stop_flag"] = True
            st.rerun()

    # Stop processing button - shown during active processing
    if st.session_state.get("_processing", False):
        if st.button("Stop", type="secondary", use_container_width=True):
            st.session_state["_processing"] = False
            st.session_state["_stop_flag"] = True
            st.rerun()

    col_in, col_v = st.columns([6, 1])
    with col_in: question = st.chat_input("Ask anything about your document...")
    with col_v:
        try: is_https = st.context.headers.get("x-forwarded-proto") == "https"
        except Exception: is_https = False
        if is_https:
            try:
                from streamlit_mic_recorder import mic_recorder
                audio = mic_recorder(start_prompt="ðŸŽ™", stop_prompt="â¹", key="voice")
                if audio and audio.get("bytes"):
                    fn = lazy("agents.voice_agent", "transcribe_audio_file")
                    vr = fn(audio["bytes"])
                    if vr["success"]: question = vr["text"]; st.success(vr["text"])
            except Exception: st.button("ðŸŽ™", help="Voice unavailable")
        else:
            if st.button("ðŸŽ™", help="Voice requires HTTPS"): st.info("Voice available on Streamlit Cloud.")

    if question: answer_q(question, deps); st.rerun()

if __name__ == "__main__":
    main()