import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import time
import io
import re
import json
from loguru import logger


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="ðŸ§ ",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    from utils.error_handler import handle_error
    return {
        "load_document": load_document,
        "chunk_text": chunk_text,
        "create_vector_store": create_vector_store,
        "create_bm25_index": create_bm25_index,
        "run_workflow": run_workflow,
        "SessionMemory": SessionMemory,
        "FileMemoryManager": FileMemoryManager,
        "validate_file": validate_file,
        "validate_question": validate_question,
        "handle_error": handle_error,
    }


def get_llm(temp=0.3):
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key not in ["your_groq_key_here", ""]:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=temp,
                max_tokens=2048
            )
        except Exception:
            pass
    from langchain_google_genai import ChatGoogleGenerativeAI
    from utils.config import get_google_api_key
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=temp
    )


def lazy(module, func):
    import importlib
    return getattr(importlib.import_module(module), func)


def init(deps):
    defaults = {
        "memory": deps["SessionMemory"](),
        "fmm": deps["FileMemoryManager"](),
        "chat": [],
        "file_path": "",
        "doc_ready": False,
        "doc_name": "",
        "doc_text": "",
        "prompts": [],
        "pending_prompts": "",
        "answer_mode": "detailed",
        "multi_docs": [],
        "active_doc_index": 0,
        "last_q": "",
        "last_a": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def detect_generation_intent(q: str) -> dict:
    ql = q.lower()
    return {
        "chart": any(w in ql for w in [
            "chart", "graph", "plot", "visual", "bar", "pie",
            "line", "graphical", "diagram", "visualize", "animate",
            "different visual", "visual representation", "show data"
        ]),
        "quiz": any(w in ql for w in [
            "quiz", "test", "mcq", "exam", "question me", "ask me"
        ]),
        "export": any(w in ql for w in [
            "download", "export", "save", "generate file",
            "create pdf", "make pdf", "create doc", "make excel",
            "generate excel", "create csv", "generate document",
            "give me a file", "regenerate", "updated document"
        ]),
        "excel": any(w in ql for w in [
            "excel", "xlsx", "spreadsheet", "generate excel",
            "create excel", "make excel", "give excel"
        ]),
        "pptx": any(w in ql for w in [
            "ppt", "powerpoint", "presentation", "slides",
            "create presentation", "make slides"
        ]),
        "table": any(w in ql for w in [
            "table", "tabular", "rows", "columns", "structured"
        ]),
    }


def gen_chart(doc_text: str, question: str):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["text", "question"],
            template="""You are a data extraction expert. Extract numerical data from this document to create a chart for: {question}

Document excerpt: {text}

Return ONLY a valid JSON object. No explanation. No markdown. No code blocks.
Example format: {{"labels": ["Category A", "Category B"], "values": [25, 75], "title": "Distribution", "type": "bar"}}
- type must be one of: bar, pie, line
- Extract real numbers from the document
- If multiple charts possible, pick the most meaningful one
- labels and values must have same length"""
        )

        llm = get_llm(temp=0)
        chain = prompt | llm
        result = chain.invoke({"text": doc_text[:3000], "question": question})
        content = result.content.strip()
        content = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()

        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
            else:
                return None

        labels = data.get("labels", [])
        values = data.get("values", [])
        title = data.get("title", "Document Data")
        ctype = data.get("type", "bar")

        if not labels or not values or len(labels) != len(values):
            return None

        values = [float(v) for v in values]
        df = pd.DataFrame({"Category": labels, "Value": values})

        COLORS = ["#6366f1", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b",
                  "#ef4444", "#ec4899", "#14b8a6", "#f97316", "#84cc16"]

        layout = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Sora, Inter, sans-serif", size=13),
            title=dict(
                text=title, font=dict(color="#e2e8f0", size=17),
                x=0.5, xanchor="center"
            ),
            margin=dict(t=60, b=50, l=50, r=30),
            height=420,
            showlegend=ctype == "pie",
        )

        if ctype == "pie":
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.4,
                marker=dict(colors=COLORS[:len(labels)],
                           line=dict(color="rgba(0,0,0,0.3)", width=2)),
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}<extra></extra>"
            ))
        elif ctype == "line":
            fig = go.Figure(go.Scatter(
                x=labels, y=values,
                mode="lines+markers",
                line=dict(color="#6366f1", width=3),
                marker=dict(size=9, color="#06b6d4",
                           line=dict(color="#fff", width=2)),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>"
            ))
            layout["xaxis"] = dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b"))
            layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b"))
        else:
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(
                    color=COLORS[:len(labels)],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    cornerradius=6,
                ),
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>"
            ))
            layout["xaxis"] = dict(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#64748b"))
            layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b"))
            layout["bargap"] = 0.3

        fig.update_layout(**layout)
        return fig

    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None


def gen_multiple_charts(doc_text: str, question: str):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["text", "question"],
            template="""Extract data for up to 3 different meaningful charts from this document for: {question}

Document: {text}

Return ONLY valid JSON array. No markdown. No explanation.
Format: [{{"labels": ["A","B"], "values": [1,2], "title": "Chart 1", "type": "bar"}}, ...]
Types: bar, pie, line. Maximum 3 charts."""
        )

        llm = get_llm(temp=0)
        result = (prompt | llm).invoke({"text": doc_text[:3000], "question": question})
        content = result.content.strip()
        content = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()

        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0:
            return []

        charts_data = json.loads(content[start:end])
        figs = []
        for cd in charts_data[:3]:
            labels = cd.get("labels", [])
            values = cd.get("values", [])
            if labels and values and len(labels) == len(values):
                fig = gen_chart_from_data(labels, [float(v) for v in values], cd.get("title", ""), cd.get("type", "bar"))
                if fig:
                    figs.append((cd.get("title", "Chart"), fig))
        return figs
    except Exception as e:
        logger.error(f"Multi-chart error: {e}")
        return []


def gen_chart_from_data(labels, values, title, ctype):
    try:
        import plotly.graph_objects as go
        COLORS = ["#6366f1", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]
        layout = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter, sans-serif", size=12),
            title=dict(text=title, font=dict(color="#e2e8f0", size=15), x=0.5),
            margin=dict(t=55, b=45, l=45, r=25), height=350,
            showlegend=ctype == "pie",
        )
        if ctype == "pie":
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.38,
                marker=dict(colors=COLORS[:len(labels)], line=dict(color="rgba(0,0,0,0.3)", width=2)),
                textinfo="percent+label"))
        elif ctype == "line":
            fig = go.Figure(go.Scatter(x=labels, y=values, mode="lines+markers",
                line=dict(color="#6366f1", width=2.5), marker=dict(size=8, color="#06b6d4"),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)"))
            layout["xaxis"] = dict(gridcolor="rgba(255,255,255,0.04)")
            layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.04)")
        else:
            fig = go.Figure(go.Bar(x=labels, y=values,
                marker=dict(color=COLORS[:len(labels)], cornerradius=5)))
            layout["xaxis"] = dict(gridcolor="rgba(255,255,255,0.03)")
            layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.04)")
            layout["bargap"] = 0.28
        fig.update_layout(**layout)
        return fig
    except Exception:
        return None


def gen_quiz(text: str) -> list:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate exactly 5 multiple choice quiz questions from this document.

Document: {text}

Use EXACTLY this format (blank line between questions):
Q: [question text]
A: [correct answer]
B: [wrong option]
C: [wrong option]
D: [wrong option]
ANSWER: A"""
        )
        llm = get_llm(temp=0.3)
        result = (prompt | llm).invoke({"text": text[:2500]})
        questions = []
        for block in result.content.strip().split("\n\n"):
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            q = {}
            opts = []
            for line in lines:
                if line.startswith("Q:"):
                    q["question"] = line[2:].strip()
                elif len(line) > 2 and line[1] == ":" and line[0] in "ABCD":
                    opts.append((line[0], line[2:].strip()))
                elif line.startswith("ANSWER:"):
                    q["answer"] = line[7:].strip()
            if q.get("question") and opts:
                q["options"] = opts
                questions.append(q)
        return questions[:5]
    except Exception:
        return []


def gen_resources(question: str, answer: str) -> dict:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""Based on this Q&A generate specific learning resources.
Q: {question}
A: {answer}

Return ONLY JSON. No explanation.
{{"youtube": ["specific search 1", "specific search 2", "specific search 3"],
  "websites": ["topic 1 to Google", "topic 2 to Google"],
  "academic": ["scholar search term 1"]}}"""
        )
        llm = get_llm(temp=0.2)
        result = (prompt | llm).invoke({"question": question, "answer": answer[:500]})
        content = result.content.strip()
        content = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0:
            return json.loads(content[start:end])
        return {}
    except Exception:
        return {}


def gen_file(content: str, fmt: str) -> bytes:
    if fmt == "txt":
        return content.encode("utf-8")
    elif fmt == "docx":
        from docx import Document
        doc = Document()
        doc.add_heading("DocuMind AI â€” Generated Report", 0)
        for line in content.split("\n"):
            if line.strip():
                if line.startswith("## "):
                    doc.add_heading(line[3:], 2)
                elif line.startswith("# "):
                    doc.add_heading(line[2:], 1)
                else:
                    doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    elif fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            for line in content.split("\n"):
                if line.strip():
                    story.append(Paragraph(
                        line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                        styles["Normal"]
                    ))
                    story.append(Spacer(1, 5))
            doc.build(story)
            return buf.getvalue()
        except Exception:
            return content.encode("utf-8")
    elif fmt == "csv":
        import pandas as pd
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        df = pd.DataFrame({"Content": lines})
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "xlsx":
        import pandas as pd
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        df = pd.DataFrame({"Content": lines})
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()
    return content.encode("utf-8")


def tts_html(text: str) -> str:
    clean = text[:500].replace('"', "").replace("'", "").replace("\n", " ").replace("<", "").replace(">", "")
    return f"""<script>
function dmS(){{window.speechSynthesis.cancel();var u=new SpeechSynthesisUtterance("{clean}");u.rate=0.9;u.pitch=1;u.volume=1;window.speechSynthesis.speak(u);}}
function dmX(){{window.speechSynthesis.cancel();}}
</script>
<div style="display:flex;gap:8px;margin-top:10px;align-items:center;">
<button onclick="dmS()" style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:8px;color:#a5b4fc;padding:6px 14px;font-size:0.74rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;transition:all 0.2s;">ðŸ”Š Read Aloud</button>
<button onclick="dmX()" style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.15);border-radius:8px;color:#fca5a5;padding:6px 14px;font-size:0.74rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;">â¹ Stop</button>
</div>"""


def gen_prompts(text: str) -> list:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate 6 specific interesting questions a user would ask about this document.
Document: {text}
Rules: specific to content, mix factual/analytical/summary, under 10 words each, one per line, no numbering.
Questions:"""
        )
        llm = get_llm(temp=0.5)
        result = (prompt | llm).invoke({"text": text[:1500]})
        lines = [l.strip() for l in result.content.strip().split("\n") if l.strip() and len(l.strip()) > 5]
        return lines[:6]
    except Exception:
        return ["What is this document about?", "Summarize the key points",
                "What are the main findings?", "What actions are recommended?",
                "What are the risks mentioned?", "Who is this document for?"]


def process_doc(f, deps) -> bool:
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    try:
        suffix = f".{f.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        ok, msg = deps["validate_file"](tmp_path)
        if not ok:
            st.error(msg)
            return False

        st.session_state.file_path = tmp_path
        st.session_state.doc_name = f.name

        info = st.empty()

        info.markdown(
            "<p style='color:#a5b4fc;font-size:0.82rem;'>?? Reading...</p>",
            unsafe_allow_html=True
        )
        text = deps["load_document"](tmp_path)
        st.session_state.doc_text = text

        info.markdown(
            "<p style='color:#a5b4fc;font-size:0.82rem;'>?? Chunking...</p>",
            unsafe_allow_html=True
        )
        chunks = deps["chunk_text"](text)

        info.markdown(
            "<p style='color:#a5b4fc;font-size:0.82rem;'>?? Indexing in parallel...</p>",
            unsafe_allow_html=True
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(deps["create_vector_store"], chunks)
            f2 = executor.submit(deps["create_bm25_index"], chunks)
            f1.result()
            f2.result()

        info.markdown(
            "<p style='color:#a5b4fc;font-size:0.82rem;'>?? Generating prompts...</p>",
            unsafe_allow_html=True
        )
        st.session_state.prompts = gen_prompts(text)
        info.empty()

        st.session_state.doc_ready = True
        st.session_state.chat = []
        st.session_state.memory = st.session_state.fmm.get_memory(f.name)

        existing = [d["name"] for d in st.session_state.multi_docs]
        if f.name not in existing:
            st.session_state.multi_docs.append({
                "name": f.name,
                "path": tmp_path,
                "text": text
            })

        return True

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def show_msg(role, content, tts=False):
    if role == "human":
        with st.chat_message("user", avatar="ðŸ‘¤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="ðŸ§ "):
            st.markdown(content)
            if tts:
                components.html(tts_html(content), height=52)


def show_resources_section(question: str, answer: str):
    with st.spinner("ðŸŒ Finding related resources..."):
        k = gen_resources(question, answer)

    if not k:
        return

    st.markdown("""
    <div style='margin-top:20px;padding:16px 0 8px;border-top:1px solid rgba(99,102,241,0.1);'>
        <span style='font-size:0.7rem;font-weight:700;color:rgba(99,102,241,0.7);text-transform:uppercase;letter-spacing:2px;'>Related Resources</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if k.get("youtube"):
            st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#fca5a5;'>ðŸ“º YouTube</span>", unsafe_allow_html=True)
            for s in k["youtube"][:3]:
                url = f"https://www.youtube.com/results?search_query={s.strip().replace(' ', '+')}"
                st.markdown(f"<div style='background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.1);border-radius:8px;padding:8px 12px;margin:5px 0;'><a href='{url}' target='_blank' style='color:#fca5a5;text-decoration:none;font-size:0.78rem;font-weight:500;'>â–¶ {s}</a></div>", unsafe_allow_html=True)

    with c2:
        if k.get("websites"):
            st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#a5b4fc;'>ðŸ” Web Search</span>", unsafe_allow_html=True)
            for t in k["websites"][:3]:
                url = f"https://www.google.com/search?q={t.strip().replace(' ', '+')}"
                st.markdown(f"<div style='background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.1);border-radius:8px;padding:8px 12px;margin:5px 0;'><a href='{url}' target='_blank' style='color:#a5b4fc;text-decoration:none;font-size:0.78rem;font-weight:500;'>ðŸ”— {t}</a></div>", unsafe_allow_html=True)

    with c3:
        if k.get("academic"):
            st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#67e8f9;'>ðŸŽ“ Academic</span>", unsafe_allow_html=True)
            for s in k["academic"][:2]:
                url = f"https://scholar.google.com/scholar?q={s.strip().replace(' ', '+')}"
                st.markdown(f"<div style='background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.1);border-radius:8px;padding:8px 12px;margin:5px 0;'><a href='{url}' target='_blank' style='color:#67e8f9;text-decoration:none;font-size:0.78rem;font-weight:500;'>ðŸ“– {s}</a></div>", unsafe_allow_html=True)


def answer_q(question: str, deps):
    ok, msg = deps["validate_question"](question)
    if not ok:
        st.warning(msg)
        return

    intent = detect_generation_intent(question)
    show_msg("human", question)
    st.session_state.chat.append({"role": "human", "content": question})

    with st.spinner("\U0001f9e0 Thinking..."):
        try:
            result = deps["run_workflow"](
                question=question,
                memory=st.session_state.memory,
                file_path=st.session_state.file_path,
                answer_mode=st.session_state.answer_mode
            )
            answer = result["answer"]
            evidence = result["evidence"]
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

    show_msg("assistant", answer, tts=True)

    if evidence:
        with st.expander("Evidence Sources"):
            for i, chunk in enumerate(evidence, 1):
                st.markdown(
                    f"<div style='background:rgba(99,102,241,0.04);"
                    f"border-left:2px solid rgba(99,102,241,0.3);"
                    f"border-radius:0 8px 8px 0;padding:9px 13px;"
                    f"margin:6px 0;font-size:0.79rem;color:#64748b;"
                    f"line-height:1.55;'>"
                    f"<strong style='color:#a5b4fc;'>Source {i}</strong>"
                    f"<br>{chunk}</div>",
                    unsafe_allow_html=True
                )

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.last_q = question
    st.session_state.last_a = answer

    if intent["chart"]:
        with st.spinner("\U0001f4ca Generating charts..."):
            charts = gen_multiple_charts(st.session_state.doc_text, question)
        if charts:
            if len(charts) == 1:
                st.plotly_chart(charts[0][1], use_container_width=True)
            elif len(charts) >= 2:
                c1, c2 = st.columns(2)
                for i, (title, fig) in enumerate(charts[:4]):
                    with (c1 if i % 2 == 0 else c2):
                        st.plotly_chart(fig, use_container_width=True)
        else:
            single = gen_chart(st.session_state.doc_text, question)
            if single:
                st.plotly_chart(single, use_container_width=True)
            else:
                st.info("No numerical data found for chart generation.")

    if intent["quiz"]:
        with st.spinner("\U0001f9e9 Generating quiz..."):
            quiz = gen_quiz(st.session_state.doc_text)
        if quiz:
            st.markdown("### \U0001f9e9 Quiz")
            for i, q in enumerate(quiz, 1):
                st.markdown(
                    f"<div style='background:rgba(139,92,246,0.05);"
                    f"border:1px solid rgba(139,92,246,0.12);"
                    f"border-radius:12px;padding:16px;margin-bottom:12px;'>"
                    f"<div style='color:#e2e8f0;font-size:0.88rem;"
                    f"font-weight:600;margin-bottom:10px;'>Q{i}. {q['question']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if q.get("options"):
                    user = st.radio(
                        f"Q{i}",
                        [f"{o[0]}. {o[1]}" for o in q["options"]],
                        key=f"qr{i}_{len(st.session_state.chat)}",
                        label_visibility="collapsed"
                    )
                    if st.button(
                        f"Check Q{i}",
                        key=f"qc{i}_{len(st.session_state.chat)}"
                    ):
                        correct = q.get("answer", "A")
                        if user and user.startswith(correct):
                            st.success("\u2705 Correct!")
                        else:
                            ct = next(
                                (f"{o[0]}. {o[1]}" for o in q["options"]
                                 if o[0] == correct), correct
                            )
                            st.error(f"\u274c Correct: {ct}")

    if intent["excel"]:
        st.markdown("### \U0001f4ca Generated Excel File")
        from tools.file_export_tool import generate_excel_from_document
        with st.spinner("Creating Excel file from document data..."):
            excel_bytes = generate_excel_from_document(
                st.session_state.doc_text, question
            )
        st.download_button(
            "\U0001f4e5 Download Excel File",
            data=excel_bytes,
            file_name="documind_generated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_excel_{len(st.session_state.chat)}"
        )

    elif intent["pptx"]:
        st.markdown("### \U0001f4ca Generated PowerPoint")
        from tools.file_export_tool import export_as_pptx
        with st.spinner("Creating PowerPoint presentation..."):
            pptx_path = export_as_pptx(
                answer,
                title=st.session_state.doc_name
            )
        with open(pptx_path, "rb") as f:
            st.download_button(
                "\U0001f4e5 Download PowerPoint",
                data=f,
                file_name="documind_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
                key=f"dl_pptx_{len(st.session_state.chat)}"
            )

    elif intent["export"]:
        st.markdown("### \U0001f4e5 Download Generated Document")
        ecols = st.columns(5)
        for i, (fmt, label, mime) in enumerate([
            ("txt", "TXT", "text/plain"),
            ("docx", "DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("pdf", "PDF", "application/pdf"),
            ("csv", "CSV", "text/csv"),
            ("xlsx", "Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]):
            with ecols[i]:
                fb = gen_file(answer, fmt)
                st.download_button(
                    f"\U0001f4e5 {label}",
                    data=fb,
                    file_name=f"documind.{fmt}",
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{fmt}_{len(st.session_state.chat)}"
                )

    show_resources_section(question, answer)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] * {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.stApp {
    background: #04060f !important;
}

/* â”€â”€ SIDEBAR â”€â”€ */
section[data-testid="stSidebar"] {
    background: rgba(4,6,15,0.99) !important;
    border-right: 1px solid rgba(99,102,241,0.08) !important;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

/* â”€â”€ FILE UPLOADER FIX â”€â”€ */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] > div > label {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: #374151 !important;
    font-size: 0.72rem !important;
}

/* â”€â”€ BUTTONS â”€â”€ */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 10px rgba(79,70,229,0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(99,102,241,0.3) !important;
}

/* â”€â”€ CHAT INPUT â”€â”€ */
.stChatInputContainer {
    background: rgba(8,10,22,0.9) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
}
.stChatInputContainer:focus-within {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.07) !important;
}
.stChatInputContainer textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
}

/* â”€â”€ CHAT MESSAGES â”€â”€ */
div[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* â”€â”€ EXPANDER â”€â”€ */
div[data-testid="stExpander"] {
    background: rgba(8,10,22,0.5) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
}

/* â”€â”€ SELECTBOX â”€â”€ */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(8,10,22,0.8) !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    border-radius: 9px !important;
}

/* â”€â”€ PROGRESS â”€â”€ */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4f46e5, #06b6d4) !important;
    border-radius: 4px !important;
}

/* â”€â”€ ALERTS â”€â”€ */
div[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 10px !important;
}
div[data-testid="stError"] {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px !important;
}
div[data-testid="stInfo"] {
    background: rgba(99,102,241,0.06) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
}

/* â”€â”€ DOWNLOAD BUTTONS â”€â”€ */
div[data-testid="stDownloadButton"] > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    box-shadow: none !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(99,102,241,0.15) !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* â”€â”€ RADIO â”€â”€ */
div[data-testid="stRadio"] label {
    background: rgba(8,10,22,0.5) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    margin: 3px 0 !important;
    color: #94a3b8 !important;
    transition: all 0.15s !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: rgba(99,102,241,0.25) !important;
}

/* â”€â”€ SPINNER â”€â”€ */
div[data-testid="stSpinner"] > div {
    border-top-color: #6366f1 !important;
}

/* â”€â”€ SCROLLBAR â”€â”€ */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }

hr { border-color: rgba(255,255,255,0.04) !important; }
</style>
"""

SIDEBAR_CSS = """
<style>
.dm-logo {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 18px 0 16px;
    border-bottom: 1px solid rgba(99,102,241,0.08);
    margin-bottom: 18px;
}
.dm-icon-wrap {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #06b6d4);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 18px rgba(79,70,229,0.4);
    flex-shrink: 0;
}
.dm-name-text {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.2px;
    line-height: 1.2;
}
.dm-sub-text {
    font-size: 0.58rem !important;
    color: rgba(99,102,241,0.75) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.sec-lbl {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    color: rgba(99,102,241,0.55) !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    display: block !important;
    margin: 14px 0 8px !important;
}
.doc-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 9px 12px;
    margin: 5px 0;
}
.doc-card-name {
    color: #a5b4fc !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-card-size {
    color: #3f4f6b !important;
    font-size: 0.67rem !important;
    margin-top: 2px;
}
.msg-c {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    margin: 6px 0;
}
.mc-n {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a5b4fc, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    display: block !important;
    line-height: 1 !important;
}
.mc-l {
    font-size: 0.58rem !important;
    color: #3f4f6b !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-top: 4px !important;
    display: block !important;
}
</style>
"""


def show_hero():
    components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }

body {
    background: #04060f;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 40px;
    text-align: center;
    overflow-x: hidden;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 55% 45% at 20% 25%, rgba(99,102,241,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 45% 35% at 80% 70%, rgba(6,182,212,0.08) 0%, transparent 55%);
    pointer-events: none;
}

@keyframes blink {
    0%,100%{opacity:1;transform:scale(1);}
    50%{opacity:0.3;transform:scale(0.8);}
}

@keyframes flow {
    0%{background-position:0% center;}
    100%{background-position:300% center;}
}

@keyframes fadeUp {
    from{opacity:0;transform:translateY(20px);}
    to{opacity:1;transform:translateY(0);}
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 100px;
    padding: 8px 22px;
    font-size: 0.65rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 32px;
    animation: fadeUp 0.6s ease forwards;
}

.dot {
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: blink 2s ease infinite;
}

h1 {
    font-size: clamp(3rem, 8vw, 6rem);
    font-weight: 900;
    letter-spacing: -4px;
    line-height: 0.95;
    margin-bottom: 22px;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 25%, #06b6d4 55%, #a5b4fc 80%, #ffffff 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: flow 6s linear infinite, fadeUp 0.8s ease forwards;
}

.subtitle {
    font-size: 1.05rem;
    color: #4b5563;
    max-width: 460px;
    line-height: 1.8;
    margin: 0 auto 52px;
    animation: fadeUp 1s ease forwards;
}

.stats {
    display: flex;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(8,10,22,0.8);
    max-width: 560px;
    margin: 0 auto 64px;
    animation: fadeUp 1.2s ease forwards;
}

.stat {
    flex: 1;
    padding: 20px 10px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.05);
}
.stat:last-child { border-right: none; }

.stat-val {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}

.stat-lbl {
    font-size: 0.58rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
    display: block;
}

.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    max-width: 900px;
    margin: 0 auto;
    animation: fadeUp 1.4s ease forwards;
}

.card {
    background: rgba(8,10,22,0.8);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: left;
    transition: all 0.3s ease;
    cursor: default;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.card:hover {
    border-color: rgba(99,102,241,0.22);
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(99,102,241,0.12);
}

.card:hover::before { opacity: 1; }

.ic { font-size: 1.5rem; margin-bottom: 10px; display: block; }
.nm { color: #e2e8f0; font-size: 0.83rem; font-weight: 600; margin-bottom: 4px; }
.tx { color: #374151; font-size: 0.71rem; line-height: 1.5; }
</style>
</head>
<body>
<div class="badge">
    <div class="dot"></div>
    Enterprise Document Intelligence
</div>

<h1>DocuMind AI</h1>

<p class="subtitle">
    Upload any document. Ask in plain English.<br>
    Get answers, charts, quizzes, and knowledge â€” instantly.
</p>

<div class="stats">
    <div class="stat">
        <span class="stat-val">6+</span>
        <span class="stat-lbl">Formats</span>
    </div>
    <div class="stat">
        <span class="stat-val">5</span>
        <span class="stat-lbl">AI Agents</span>
    </div>
    <div class="stat">
        <span class="stat-val">20+</span>
        <span class="stat-lbl">Features</span>
    </div>
    <div class="stat">
        <span class="stat-val">100%</span>
        <span class="stat-lbl">Private</span>
    </div>
</div>

<div class="grid">
    <div class="card">
        <span class="ic">ðŸ“Š</span>
        <div class="nm">Auto Charts</div>
        <div class="tx">Say "show chart" â€” real Plotly graphs generated instantly</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ§©</span>
        <div class="nm">Quiz Mode</div>
        <div class="tx">Say "give me a quiz" â€” MCQ questions from document</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ”Š</span>
        <div class="nm">Read Aloud</div>
        <div class="tx">Every answer spoken aloud by browser TTS</div>
    </div>
    <div class="card">
        <span class="ic">ðŸŒ</span>
        <div class="nm">Live Resources</div>
        <div class="tx">YouTube + Google + Scholar links after every answer</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ“¥</span>
        <div class="nm">Export Anything</div>
        <div class="tx">PDF, DOCX, Excel, CSV â€” just ask to download</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ’¡</span>
        <div class="nm">Smart Prompts</div>
        <div class="tx">Auto-generated questions from your document</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ”</span>
        <div class="nm">Hybrid Search</div>
        <div class="tx">FAISS semantic + BM25 keyword with RRF reranking</div>
    </div>
    <div class="card">
        <span class="ic">ðŸ”’</span>
        <div class="nm">100% Private</div>
        <div class="tx">Documents never leave your machine â€” ever</div>
    </div>
</div>
</body>
</html>
    """, height=900, scrolling=False)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    deps = load_core()
    init(deps)
    if st.session_state.get("pending_prompts") and not st.session_state.prompts:
        st.session_state.prompts = gen_prompts(st.session_state.pending_prompts)
        st.session_state.pending_prompts = ""

    with st.sidebar:
        st.markdown("""
        <div class='dm-logo'>
            <div class='dm-icon-wrap'>ðŸ§ </div>
            <div>
                <div class='dm-name-text'>DocuMind AI</div>
                <div class='dm-sub-text'>Document Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<span class='sec-lbl'>Upload Document</span>", unsafe_allow_html=True)
        f = st.file_uploader(
            " ",
            type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
            label_visibility="hidden"
        )

        if f:
            kb = f.size / 1024
            st.markdown(f"""
            <div class='doc-card'>
                <div class='doc-card-name'>ðŸ“„ {f.name}</div>
                <div class='doc-card-size'>{kb:.0f} KB</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Process Document â†’", use_container_width=True):
                if process_doc(f, deps):
                    st.success("âœ… Document ready")
                    st.rerun()

        if st.session_state.doc_ready:
            st.markdown("---")
            st.markdown(
                "<span class='sec-lbl'>Add Another Document</span>",
                unsafe_allow_html=True
            )
            f2 = st.file_uploader(
                " ",
                type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
                label_visibility="hidden",
                key="second_uploader"
            )
            if f2:
                kb2 = f2.size / 1024
                st.markdown(
                    f"<div class='doc-card'><div class='doc-card-name'>ðŸ“„ {f2.name}</div>"
                    f"<div class='doc-card-size'>{kb2:.0f} KB</div></div>",
                    unsafe_allow_html=True
                )
                if st.button("Add Document â†’", use_container_width=True, key="add_doc2"):
                    existing = [d["name"] for d in st.session_state.multi_docs]
                    if f2.name not in existing:
                        with st.spinner("Processing additional document..."):
                            success = process_doc(f2, deps)
                        if success:
                            st.success(f"âœ… {f2.name} added!")
                            st.rerun()
                    else:
                        st.info("This document is already loaded.")

        if st.session_state.doc_ready:
            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Answer Style</span>", unsafe_allow_html=True)
            mode = st.selectbox(
                "m",
                options=["detailed", "quick", "bullet", "beginner", "executive", "table"],
                format_func=lambda x: {
                    "detailed": "ðŸ“ Detailed",
                    "quick": "âš¡ Quick",
                    "bullet": "â€¢ Bullets",
                    "beginner": "ðŸŽ“ Beginner",
                    "executive": "ðŸ’¼ Executive",
                    "table": "ðŸ“Š Table"
                }[x],
                label_visibility="collapsed"
            )
            st.session_state.answer_mode = mode

            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Active Document</span>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='doc-card'>
                <div class='doc-card-name'>ðŸ“„ {st.session_state.doc_name}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<span class='sec-lbl'>Session</span>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='msg-c'>
                <span class='mc-n'>{len(st.session_state.chat)}</span>
                <span class='mc-l'>Messages</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<span class='sec-lbl'>Quick Actions</span>", unsafe_allow_html=True)

            for act in [
                "ðŸ“ Summarize document",
                "âœ… Extract action items",
                "âš ï¸ Identify all risks",
                "ðŸ“Š Show data as charts",
                "â“ Generate FAQ",
                "ðŸ§© Create a quiz"
            ]:
                if st.button(act, use_container_width=True, key=f"qa{act[:6]}"):
                    answer_q(act, deps)
                    st.rerun()

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("ðŸ—‘ï¸ Clear", use_container_width=True):
                    st.session_state.memory.clear()
                    st.session_state.chat = []
                    st.rerun()
            with c2:
                if st.button("ðŸ“‚ New", use_container_width=True):
                    st.session_state.doc_ready = False
                    st.session_state.file_path = ""
                    st.session_state.doc_name = ""
                    st.session_state.doc_text = ""
                    st.session_state.chat = []
                    st.session_state.prompts = []
                    st.session_state.memory = deps["SessionMemory"]()
                    st.rerun()

    if not st.session_state.doc_ready:
        show_hero()
        return

    if len(st.session_state.multi_docs) > 0:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.12);
        border-radius:12px;padding:16px;margin-bottom:16px;'>
        <div style='color:rgba(99,102,241,0.7);font-size:0.62rem;font-weight:700;
        text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;'>
        Uploaded Documents
        </div>
        </div>
        """, unsafe_allow_html=True)

        for i, doc in enumerate(st.session_state.multi_docs):
            col_doc, col_sel = st.columns([4, 1])
            with col_doc:
                is_active = doc["name"] == st.session_state.doc_name
                color = "#3fb950" if is_active else "#94a3b8"
                indicator = "â—" if is_active else "â—‹"
                st.markdown(
                    f"<div style='color:{color};font-size:0.8rem;padding:6px 0;'>"
                    f"{indicator} ðŸ“„ {doc['name']}</div>",
                    unsafe_allow_html=True
                )
            with col_sel:
                if not is_active:
                    if st.button("Switch", key=f"sw_{i}"):
                        st.session_state.active_doc_index = i
                        st.session_state.doc_name = doc["name"]
                        st.session_state.file_path = doc["path"]
                        st.session_state.doc_text = doc["text"]
                        st.session_state.memory = (
                            st.session_state.fmm.get_memory(doc["name"])
                        )
                        st.rerun()

        if len(st.session_state.multi_docs) > 1:
            st.markdown("---")
            st.markdown(
                "<p style='color:rgba(99,102,241,0.6);font-size:0.7rem;font-weight:700;"
                "text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>"
                "Multi-Document Query</p>",
                unsafe_allow_html=True
            )

            mq = st.text_input(
                "Ask across all documents:",
                placeholder="e.g. Compare all documents, Find common themes...",
                key="multi_q"
            )

            col_mq1, col_mq2 = st.columns(2)
            with col_mq1:
                if st.button("ðŸ” Query All Documents", use_container_width=True):
                    if mq:
                        with st.spinner(f"Querying {len(st.session_state.multi_docs)} documents..."):
                            fn = lazy("agents.multi_document_agent", "query_multiple_documents")
                            ans = fn(mq, st.session_state.multi_docs)
                        show_msg("assistant", ans, tts=True)
                        st.session_state.chat.append({
                            "role": "assistant",
                            "content": f"**Multi-Document Answer:**\n\n{ans}"
                        })

            with col_mq2:
                if st.button("ðŸ“Š Compare All Documents", use_container_width=True):
                    with st.spinner("Comparing all documents..."):
                        fn = lazy("agents.multi_document_agent", "compare_documents")
                        comparison = fn(st.session_state.multi_docs)
                    show_msg("assistant", comparison, tts=True)
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": f"**Document Comparison:**\n\n{comparison}"
                    })

        st.markdown("---")

    if st.session_state.prompts:
        st.markdown(
            "<p style='color:rgba(99,102,241,0.6);font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;'>ðŸ’¡ Suggested from your document</p>",
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        for i, p in enumerate(st.session_state.prompts):
            with cols[i % 3]:
                if st.button(p, key=f"sp{i}", use_container_width=True):
                    answer_q(p, deps)

        st.markdown("---")

    for msg in st.session_state.chat:
        show_msg(msg["role"], msg["content"])

    col_in, col_v = st.columns([6, 1])
    with col_in:
        question = st.chat_input(
            "Ask anything â€” type 'show chart', 'give me a quiz', 'download as pdf' for special features..."
        )
    with col_v:
        try:
            is_https = st.context.headers.get("x-forwarded-proto") == "https"
        except Exception:
            is_https = False
        if is_https:
            try:
                from streamlit_mic_recorder import mic_recorder
                audio = mic_recorder(start_prompt="ðŸŽ¤", stop_prompt="â¹ï¸", key="voice")
                if audio and audio.get("bytes"):
                    fn = lazy("agents.voice_agent", "transcribe_audio_file")
                    vr = fn(audio["bytes"])
                    if vr["success"]:
                        question = vr["text"]
                        st.success(f"ðŸŽ¤ {question}")
            except Exception:
                st.button("ðŸŽ¤", help="Voice unavailable")
        else:
            if st.button("ðŸŽ¤", help="Voice works after HTTPS deployment"):
                st.info("ðŸŽ¤ Voice works on Streamlit Cloud")

    if question:
        answer_q(question, deps)


if __name__ == "__main__":
    main()

