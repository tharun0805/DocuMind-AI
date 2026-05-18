import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import time
import io
import re
from loguru import logger


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><defs><linearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'><stop offset='0%25' stop-color='%234f46e5'/><stop offset='100%25' stop-color='%2306b6d4'/></linearGradient></defs><rect width='100' height='100' rx='22' fill='url(%23g)'/><text y='.9em' font-size='72' x='12'>🧠</text></svg>",
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
    import os
    from dotenv import load_dotenv
    load_dotenv()
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
        "entities": {},
        "insights": {},
        "doc_map": [],
        "quiz": [],
        "prompts": [],
        "answer_mode": "detailed",
        "multi_docs": [],
        "last_q": "",
        "last_a": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def detect_intent(question: str) -> dict:
    q = question.lower()
    return {
        "wants_chart": any(w in q for w in ["chart", "graph", "plot", "visual", "diagram", "bar", "pie", "line", "show me", "graphical", "animate"]),
        "wants_table": any(w in q for w in ["table", "tabular", "spreadsheet", "rows", "columns"]),
        "wants_quiz": any(w in q for w in ["quiz", "test", "question", "mcq", "exam"]),
        "wants_export": any(w in q for w in ["download", "export", "save", "generate file", "create pdf", "create doc", "make excel"]),
        "wants_resources": any(w in q for w in ["resource", "learn", "youtube", "video", "website", "link", "reference", "more about", "study"]),
        "wants_summary": any(w in q for w in ["summarize", "summary", "overview", "brief", "outline", "tldr"]),
        "wants_entities": any(w in q for w in ["who", "people", "person", "organization", "company", "date", "location", "where", "when", "entities"]),
    }


def gen_chart_auto(text: str, question: str):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["text", "question"],
            template="""Analyze this document and extract numerical/categorical data to create a meaningful chart for: {question}

Document: {text}

Return ONLY a valid Python dict. No explanation. No markdown. No backticks.
Format: {{"labels": ["A","B","C"], "values": [10, 20, 30], "title": "Title Here", "type": "bar", "xlabel": "Category", "ylabel": "Count"}}
type options: bar, pie, line, scatter
Make the data meaningful and accurate from the document."""
        )

        llm = get_llm(temp=0)
        result = (prompt | llm).invoke({"text": text[:3000], "question": question})
        content = result.content.strip()
        content = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()

        data = eval(content)
        labels = data.get("labels", [])
        values = data.get("values", [])
        title = data.get("title", "Document Analysis")
        chart_type = data.get("type", "bar")
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")

        if not labels or not values:
            return None

        df = pd.DataFrame({"Label": labels, "Value": values})

        colors = ["#4f46e5", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

        if chart_type == "pie":
            fig = px.pie(
                df, names="Label", values="Value", title=title,
                color_discrete_sequence=colors,
                hole=0.35
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
        elif chart_type == "line":
            fig = px.line(
                df, x="Label", y="Value", title=title,
                markers=True, color_discrete_sequence=["#4f46e5"],
                labels={"Label": xlabel, "Value": ylabel}
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
        elif chart_type == "scatter":
            fig = px.scatter(
                df, x="Label", y="Value", title=title,
                color_discrete_sequence=["#4f46e5"],
                labels={"Label": xlabel, "Value": ylabel}
            )
        else:
            fig = px.bar(
                df, x="Label", y="Value", title=title,
                color="Label", color_discrete_sequence=colors,
                labels={"Label": xlabel, "Value": ylabel}
            )
            fig.update_traces(marker_line_width=0)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            title=dict(font=dict(color="#e2e8f0", size=16, family="Inter"), x=0.5),
            showlegend=chart_type == "pie",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
            margin=dict(t=60, b=40, l=40, r=40),
            height=400
        )

        return fig
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None


def gen_quiz_auto(text: str) -> list:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate exactly 5 multiple choice quiz questions from this document.

Document: {text}

Use EXACTLY this format for each question (blank line between questions):
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
                elif line[:2] in ["A:", "B:", "C:", "D:"]:
                    opts.append((line[0], line[2:].strip()))
                elif line.startswith("ANSWER:"):
                    q["answer"] = line[7:].strip()
            if q.get("question") and opts:
                q["options"] = opts
                questions.append(q)
        return questions[:5]
    except Exception:
        return []


def gen_resources_auto(question: str, answer: str) -> dict:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""Based on this Q&A suggest learning resources.
Q: {question}
A: {answer}

YOUTUBE: [3 specific YouTube search queries, one per line]
TOPICS: [3 related topics to explore, one per line]
SCHOLAR: [2 academic search terms, one per line]"""
        )
        llm = get_llm(temp=0.3)
        result = (prompt | llm).invoke({"question": question, "answer": answer})
        content = result.content.strip()
        out = {}
        current = None
        items = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            for key in ["YOUTUBE", "TOPICS", "SCHOLAR"]:
                if line.startswith(key + ":"):
                    if current:
                        out[current] = items
                    current = key
                    rest = line[len(key)+1:].strip()
                    items = [rest] if rest else []
                    break
            else:
                if current and line:
                    items.append(line)
        if current:
            out[current] = items
        return out
    except Exception:
        return {}


def gen_file(content: str, fmt: str) -> bytes:
    if fmt == "txt":
        return content.encode("utf-8")
    elif fmt == "docx":
        from docx import Document
        doc = Document()
        doc.add_heading("DocuMind AI", 0)
        for line in content.split("\n"):
            if line.strip():
                if line.startswith("# "):
                    doc.add_heading(line[2:], 1)
                elif line.startswith("## "):
                    doc.add_heading(line[3:], 2)
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
                    story.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;"), styles["Normal"]))
                    story.append(Spacer(1, 6))
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
        df.to_excel(buf, index=False)
        return buf.getvalue()
    return content.encode("utf-8")


def tts_html(text: str) -> str:
    clean = text[:500].replace('"', "").replace("'", "").replace("\n", " ")
    return f"""<script>
function dmS(){{window.speechSynthesis.cancel();var u=new SpeechSynthesisUtterance("{clean}");u.rate=0.92;u.pitch=1;window.speechSynthesis.speak(u);}}
function dmX(){{window.speechSynthesis.cancel();}}
</script>
<div style="display:flex;gap:8px;margin-top:8px;">
<button onclick="dmS()" style="background:rgba(79,70,229,0.12);border:1px solid rgba(79,70,229,0.25);border-radius:7px;color:#a5b4fc;padding:5px 13px;font-size:0.73rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;transition:all 0.2s;">🔊 Read Aloud</button>
<button onclick="dmX()" style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.18);border-radius:7px;color:#fca5a5;padding:5px 13px;font-size:0.73rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;">⏹ Stop</button>
</div>"""


def gen_prompts(text: str) -> list:
    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate exactly 6 specific interesting questions a user would ask about this document.
Document: {text}
Rules: specific to content, mix factual/analytical/summary, under 10 words each, one per line, no numbering, no bullets.
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

        pb = st.progress(0, "📖 Reading...")
        text = deps["load_document"](tmp_path)
        st.session_state.doc_text = text

        pb.progress(35, "✂️ Chunking...")
        chunks = deps["chunk_text"](text)

        pb.progress(62, "🔢 Indexing...")
        deps["create_vector_store"](chunks)
        deps["create_bm25_index"](chunks)

        pb.progress(88, "💡 Generating prompts...")
        st.session_state.prompts = gen_prompts(text)

        pb.progress(100, "✅ Ready")
        time.sleep(0.2)
        pb.empty()

        st.session_state.doc_ready = True
        st.session_state.chat = []
        st.session_state.memory = st.session_state.fmm.get_memory(f.name)

        existing = [d["name"] for d in st.session_state.multi_docs]
        if f.name not in existing:
            st.session_state.multi_docs.append({"name": f.name, "path": tmp_path, "text": text})

        return True
    except Exception as e:
        st.error(deps["handle_error"](e, "upload"))
        return False


def show_msg(role, content, tts=False):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)
            if tts:
                components.html(tts_html(content), height=48)


def answer_q(question: str, deps):
    ok, msg = deps["validate_question"](question)
    if not ok:
        st.warning(msg)
        return

    intent = detect_intent(question)

    show_msg("human", question)
    st.session_state.chat.append({"role": "human", "content": question})

    with st.spinner("🧠 Analysing..."):
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
            st.error(deps["handle_error"](e, "workflow"))
            return

    show_msg("assistant", answer, tts=True)

    if evidence:
        with st.expander("📎 Evidence Sources"):
            for i, chunk in enumerate(evidence, 1):
                st.markdown(
                    f"<div style='background:rgba(79,70,229,0.05);border-left:2px solid rgba(79,70,229,0.4);border-radius:0 8px 8px 0;padding:9px 13px;margin:6px 0;font-size:0.79rem;color:#64748b;line-height:1.55;'><strong style='color:#a5b4fc;'>Source {i}</strong><br>{chunk}</div>",
                    unsafe_allow_html=True
                )

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.last_q = question
    st.session_state.last_a = answer

    if intent["wants_chart"]:
        with st.spinner("📊 Generating chart..."):
            fig = gen_chart_auto(st.session_state.doc_text, question)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not extract numerical data for a chart from this document.")

    if intent["wants_quiz"]:
        with st.spinner("🧩 Generating quiz..."):
            quiz = gen_quiz_auto(st.session_state.doc_text)
        if quiz:
            st.markdown("### 🧩 Quiz")
            for i, q in enumerate(quiz, 1):
                st.markdown(
                    f"<div style='background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.12);border-radius:12px;padding:16px;margin-bottom:12px;'><div style='color:#e2e8f0;font-size:0.88rem;font-weight:600;margin-bottom:10px;'>Q{i}. {q['question']}</div></div>",
                    unsafe_allow_html=True
                )
                if q.get("options"):
                    user = st.radio(f"Q{i}", [f"{o[0]}. {o[1]}" for o in q["options"]], key=f"qr{i}_{len(st.session_state.chat)}", label_visibility="collapsed")
                    if st.button(f"Check Q{i}", key=f"qc{i}_{len(st.session_state.chat)}"):
                        correct = q.get("answer", "A")
                        if user and user.startswith(correct):
                            st.success("✅ Correct!")
                        else:
                            ct = next((f"{o[0]}. {o[1]}" for o in q["options"] if o[0] == correct), correct)
                            st.error(f"❌ Correct: {ct}")

    if intent["wants_resources"] or intent["wants_chart"] or intent["wants_summary"]:
        with st.spinner("🌐 Finding resources..."):
            k = gen_resources_auto(question, answer)
        if k:
            st.markdown("### 🌐 Related Resources")
            c1, c2 = st.columns(2)
            with c1:
                if k.get("YOUTUBE"):
                    st.markdown("**📺 YouTube**")
                    for s in k["YOUTUBE"]:
                        url = f"https://www.youtube.com/results?search_query={s.strip().replace(' ', '+')}"
                        st.markdown(
                            f"<div style='background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.12);border-radius:9px;padding:10px 13px;margin-bottom:7px;transition:all 0.2s;'><a href='{url}' target='_blank' style='color:#fca5a5;text-decoration:none;font-size:0.82rem;font-weight:500;'>▶ {s}</a></div>",
                            unsafe_allow_html=True
                        )
            with c2:
                if k.get("TOPICS"):
                    st.markdown("**🔗 Explore More**")
                    for t in k["TOPICS"]:
                        url = f"https://www.google.com/search?q={t.strip().replace(' ', '+')}"
                        st.markdown(
                            f"<div style='background:rgba(79,70,229,0.05);border:1px solid rgba(79,70,229,0.1);border-radius:9px;padding:10px 13px;margin-bottom:7px;'><a href='{url}' target='_blank' style='color:#a5b4fc;text-decoration:none;font-size:0.82rem;font-weight:500;'>🔍 {t}</a></div>",
                            unsafe_allow_html=True
                        )
                if k.get("SCHOLAR"):
                    st.markdown("**🎓 Academic**")
                    for s in k["SCHOLAR"]:
                        url = f"https://scholar.google.com/scholar?q={s.strip().replace(' ', '+')}"
                        st.markdown(
                            f"<div style='background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.1);border-radius:9px;padding:10px 13px;margin-bottom:7px;'><a href='{url}' target='_blank' style='color:#67e8f9;text-decoration:none;font-size:0.82rem;font-weight:500;'>📖 {s}</a></div>",
                            unsafe_allow_html=True
                        )

    if intent["wants_export"]:
        st.markdown("### 📥 Export Answer")
        cols = st.columns(5)
        for i, (fmt, label, mime) in enumerate([
            ("txt", "TXT", "text/plain"),
            ("docx", "DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("pdf", "PDF", "application/pdf"),
            ("csv", "CSV", "text/csv"),
            ("xlsx", "XLSX", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]):
            with cols[i]:
                fb = gen_file(answer, fmt)
                st.download_button(
                    f"📥 {label}",
                    data=fb,
                    file_name=f"answer.{fmt}",
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{fmt}_{len(st.session_state.chat)}"
                )

    if intent["wants_entities"]:
        with st.spinner("🔍 Extracting entities..."):
            fn = lazy("agents.entity_agent", "extract_entities")
            entities = fn(st.session_state.doc_text[:3000])
        if entities:
            st.markdown("### 📊 Entities Found")
            icons = {"PEOPLE": "👤", "ORGANIZATIONS": "🏢", "DATES": "📅", "LOCATIONS": "📍", "KEY_TERMS": "🔑"}
            for et, vals in entities.items():
                if vals:
                    ic = icons.get(et, "•")
                    st.markdown(f"**{ic} {et}**")
                    tags = "".join([f"<span style='display:inline-block;background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.15);border-radius:6px;padding:3px 10px;margin:3px;font-size:0.76rem;color:#22d3ee;'>{v}</span>" for v in vals])
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown("")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {
    font-family: 'Sora', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #04060f;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 15% 20%, rgba(79,70,229,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 85% 75%, rgba(6,182,212,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 30% 30% at 50% 50%, rgba(139,92,246,0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

section[data-testid="stSidebar"] {
    background: rgba(4,6,15,0.98) !important;
    border-right: 1px solid rgba(79,70,229,0.08) !important;
    width: 272px !important;
}

section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

.dm-logo-wrap {
    padding: 22px 0 20px;
    border-bottom: 1px solid rgba(79,70,229,0.08);
    margin-bottom: 20px;
}

.dm-logo-inner {
    display: flex;
    align-items: center;
    gap: 11px;
}

.dm-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 0 20px rgba(79,70,229,0.35), inset 0 1px 0 rgba(255,255,255,0.1);
    flex-shrink: 0;
    position: relative;
}

.dm-icon::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
}

.dm-name {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.2px;
    line-height: 1.15;
}

.dm-sub {
    font-size: 0.6rem !important;
    color: rgba(79,70,229,0.8) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.sb-sec {
    margin-bottom: 6px;
}

.sb-lbl {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    color: rgba(79,70,229,0.6) !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    display: block !important;
    margin-bottom: 8px !important;
}

.doc-pill {
    background: rgba(79,70,229,0.06);
    border: 1px solid rgba(79,70,229,0.12);
    border-radius: 10px;
    padding: 9px 12px;
    margin: 6px 0;
}

.doc-pill-name {
    color: #a5b4fc !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    line-height: 1.3;
}

.doc-pill-size {
    color: #475569 !important;
    font-size: 0.68rem !important;
    margin-top: 2px;
}

.msg-count {
    background: rgba(79,70,229,0.06);
    border: 1px solid rgba(79,70,229,0.1);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}

.mc-num {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a5b4fc, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    display: block !important;
    line-height: 1.1 !important;
}

.mc-lbl {
    font-size: 0.6rem !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-top: 3px !important;
    display: block !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(79,70,229,0.22) !important;
    letter-spacing: 0.1px !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.3) !important;
}

.hero {
    min-height: 94vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 32px;
    text-align: center;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(79,70,229,0.07);
    border: 1px solid rgba(79,70,229,0.18);
    border-radius: 100px;
    padding: 7px 20px;
    font-size: 0.66rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 28px;
}

.hb-dot {
    width: 5px; height: 5px;
    background: #22c55e;
    border-radius: 50%;
    animation: blink 2s ease infinite;
}

@keyframes blink {
    0%,100%{opacity:1;transform:scale(1);}
    50%{opacity:0.35;transform:scale(0.75);}
}

.hero-title {
    font-size: 4.8rem;
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 30%, #06b6d4 60%, #fff 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleFlow 7s linear infinite;
}

@keyframes titleFlow {
    0%{background-position:0% center;}
    100%{background-position:250% center;}
}

.hero-p {
    font-size: 1.05rem;
    color: #475569;
    max-width: 480px;
    line-height: 1.8;
    margin: 0 auto 48px;
    font-weight: 400;
}

.hero-stats {
    display: flex;
    gap: 0;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    overflow: hidden;
    max-width: 560px;
    margin: 0 auto 64px;
    background: rgba(8,10,20,0.7);
    backdrop-filter: blur(20px);
}

.hs {
    flex: 1;
    padding: 18px 12px;
    border-right: 1px solid rgba(255,255,255,0.05);
    text-align: center;
}

.hs:last-child { border-right: none; }

.hs-v {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}

.hs-l {
    font-size: 0.6rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
    display: block;
}

.cap-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    max-width: 860px;
    margin: 0 auto;
}

.cap {
    background: rgba(8,10,20,0.7);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 13px;
    padding: 18px 14px;
    text-align: left;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.cap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,70,229,0.4), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}

.cap:hover {
    border-color: rgba(79,70,229,0.18);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(79,70,229,0.1);
}

.cap:hover::before { opacity: 1; }

.cap-ic { font-size: 1.45rem; margin-bottom: 8px; display: block; }
.cap-nm { color: #cbd5e1; font-size: 0.82rem; font-weight: 600; margin-bottom: 3px; }
.cap-tx { color: #334155; font-size: 0.71rem; line-height: 1.5; }

.stChatInputContainer {
    background: rgba(8,10,20,0.85) !important;
    border: 1px solid rgba(79,70,229,0.14) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
}

.stChatInputContainer:focus-within {
    border-color: rgba(79,70,229,0.35) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.06) !important;
}

div[data-testid="stExpander"] {
    background: rgba(8,10,20,0.5) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 11px !important;
}

.stSelectbox > div > div {
    background: rgba(8,10,20,0.8) !important;
    border: 1px solid rgba(79,70,229,0.12) !important;
    border-radius: 9px !important;
}

.stFileUploader {
    border: 2px dashed rgba(79,70,229,0.16) !important;
    border-radius: 11px !important;
    background: rgba(79,70,229,0.02) !important;
    transition: all 0.2s !important;
}

.stFileUploader:hover {
    border-color: rgba(79,70,229,0.32) !important;
    background: rgba(79,70,229,0.04) !important;
}

.prompt-btn {
    background: rgba(79,70,229,0.07) !important;
    border: 1px solid rgba(79,70,229,0.14) !important;
    border-radius: 100px !important;
    color: #a5b4fc !important;
    font-size: 0.77rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
}

.prompt-btn:hover {
    background: rgba(79,70,229,0.13) !important;
    border-color: rgba(79,70,229,0.28) !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79,70,229,0.18); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,70,229,0.32); }

hr { border-color: rgba(255,255,255,0.04) !important; }

.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #06b6d4) !important; border-radius: 4px !important; }
</style>
"""


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    deps = load_core()
    init(deps)

    with st.sidebar:
        st.markdown("""
        <div class='dm-logo-wrap'>
            <div class='dm-logo-inner'>
                <div class='dm-icon'>🧠</div>
                <div>
                    <div class='dm-name'>DocuMind AI</div>
                    <div class='dm-sub'>Document Intelligence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<span class='sb-lbl'>Upload Document</span>", unsafe_allow_html=True)
        f = st.file_uploader("f", type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"], label_visibility="collapsed")

        if f:
            kb = f.size / 1024
            st.markdown(f"<div class='doc-pill'><div class='doc-pill-name'>📄 {f.name}</div><div class='doc-pill-size'>{kb:.0f} KB</div></div>", unsafe_allow_html=True)
            if st.button("Process Document →", use_container_width=True):
                if process_doc(f, deps):
                    st.success("✅ Ready")
                    st.rerun()

        if st.session_state.doc_ready:
            st.markdown("---")
            st.markdown("<span class='sb-lbl'>Answer Style</span>", unsafe_allow_html=True)
            mode = st.selectbox("m", ["detailed", "quick", "bullet", "beginner", "executive", "table"],
                format_func=lambda x: {"detailed": "📝 Detailed", "quick": "⚡ Quick", "bullet": "• Bullets",
                "beginner": "🎓 Beginner", "executive": "💼 Executive", "table": "📊 Table"}[x],
                label_visibility="collapsed")
            st.session_state.answer_mode = mode

            st.markdown("---")
            st.markdown("<span class='sb-lbl'>Active File</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='doc-pill'><div class='doc-pill-name'>📄 {st.session_state.doc_name}</div></div>", unsafe_allow_html=True)

            st.markdown("<span class='sb-lbl'>Session</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='msg-count'><span class='mc-num'>{len(st.session_state.chat)}</span><span class='mc-lbl'>Messages</span></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<span class='sb-lbl'>Quick Actions</span>", unsafe_allow_html=True)
            for act in ["📝 Summarize document", "✅ Extract action items", "⚠️ Identify risks", "📊 Extract metrics", "❓ Generate FAQ"]:
                if st.button(act, use_container_width=True, key=f"qa{act[:5]}"):
                    with st.spinner("Working..."):
                        fn = lazy("agents.document_action_agent", "perform_document_action")
                        r = fn(action=act, context=st.session_state.doc_text[:4000], file_name=st.session_state.doc_name)
                    if r["success"]:
                        st.session_state.chat.append({"role": "assistant", "content": f"**{act}**\n\n{r['result']}"})
                        st.rerun()

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.memory.clear()
                    st.session_state.chat = []
                    st.rerun()
            with c2:
                if st.button("📂 New", use_container_width=True):
                    for k in ["doc_ready"]:
                        st.session_state[k] = False
                    for k in ["file_path", "doc_name", "doc_text"]:
                        st.session_state[k] = ""
                    st.session_state.chat = []
                    st.session_state.prompts = []
                    st.session_state.memory = deps["SessionMemory"]()
                    st.rerun()

    if not st.session_state.doc_ready:
        st.markdown("""
        <div class='hero'>
            <div class='hero-badge'><div class='hb-dot'></div>AI-Powered · Private · Instant</div>
            <div class='hero-title'>DocuMind AI</div>
            <div class='hero-p'>Upload any document. Ask anything. Get expert answers, live charts, quizzes, and knowledge links — instantly.</div>
            <div class='hero-stats'>
                <div class='hs'><span class='hs-v'>6+</span><span class='hs-l'>Formats</span></div>
                <div class='hs'><span class='hs-v'>5</span><span class='hs-l'>AI Agents</span></div>
                <div class='hs'><span class='hs-v'>20+</span><span class='hs-l'>Features</span></div>
                <div class='hs'><span class='hs-v'>100%</span><span class='hs-l'>Private</span></div>
            </div>
            <div class='cap-grid'>
                <div class='cap'><span class='cap-ic'>📊</span><div class='cap-nm'>Auto Charts</div><div class='cap-tx'>Just ask for a graph — gets generated instantly</div></div>
                <div class='cap'><span class='cap-ic'>🧩</span><div class='cap-nm'>Quiz Mode</div><div class='cap-tx'>Ask for a quiz — 5 MCQ questions generated</div></div>
                <div class='cap'><span class='cap-ic'>🔊</span><div class='cap-nm'>Read Aloud</div><div class='cap-tx'>Every answer can be spoken aloud by AI</div></div>
                <div class='cap'><span class='cap-ic'>🌐</span><div class='cap-nm'>Web Resources</div><div class='cap-tx'>Real YouTube, Google and Scholar links</div></div>
                <div class='cap'><span class='cap-ic'>📥</span><div class='cap-nm'>Export Anything</div><div class='cap-tx'>Download answers as PDF, DOCX, CSV, XLSX</div></div>
                <div class='cap'><span class='cap-ic'>💡</span><div class='cap-nm'>Smart Prompts</div><div class='cap-tx'>Auto-generated questions from your document</div></div>
                <div class='cap'><span class='cap-ic'>🔍</span><div class='cap-nm'>Hybrid Search</div><div class='cap-tx'>FAISS semantic + BM25 keyword with RRF</div></div>
                <div class='cap'><span class='cap-ic'>🔒</span><div class='cap-nm'>100% Private</div><div class='cap-tx'>Nothing ever leaves your machine</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if st.session_state.prompts:
        st.markdown("<p style='color:#475569;font-size:0.76rem;margin-bottom:8px;'>💡 Suggested questions from your document:</p>", unsafe_allow_html=True)
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
        question = st.chat_input("Ask anything — charts, quiz, summary, resources will appear automatically...")
    with col_v:
        try:
            is_https = st.context.headers.get("x-forwarded-proto") == "https"
        except Exception:
            is_https = False
        if is_https:
            try:
                from streamlit_mic_recorder import mic_recorder
                audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key="voice")
                if audio and audio.get("bytes"):
                    fn = lazy("agents.voice_agent", "transcribe_audio_file")
                    vr = fn(audio["bytes"])
                    if vr["success"]:
                        question = vr["text"]
            except Exception:
                st.button("🎤", help="Voice unavailable")
        else:
            if st.button("🎤", help="Voice works after HTTPS deployment"):
                st.info("🎤 Voice works on Streamlit Cloud")

    if question:
        answer_q(question, deps)


if __name__ == "__main__":
    main()