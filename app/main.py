import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import time
import io
from loguru import logger


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
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


def lazy(module: str, func: str):
    import importlib
    m = importlib.import_module(module)
    return getattr(m, func)


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
        "ins_loaded": False,
        "ent_loaded": False,
        "map_loaded": False,
        "quiz_loaded": False,
        "last_q": "",
        "last_a": "",
        "show_voice": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #060810;
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: #080b14 !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
    width: 280px !important;
}

section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

/* ── LOGO ── */
.dm-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 24px;
}

.dm-logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    flex-shrink: 0;
}

.dm-logo-text {
    display: flex;
    flex-direction: column;
}

.dm-logo-name {
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.dm-logo-tag {
    font-size: 0.62rem !important;
    color: #6366f1 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── SIDEBAR SECTIONS ── */
.sb-label {
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    color: rgba(99,102,241,0.7) !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    margin: 16px 0 8px !important;
    display: block !important;
}

.file-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 6px 0;
}

.file-name {
    color: #a5b4fc !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

.file-size {
    color: #64748b !important;
    font-size: 0.7rem !important;
}

.msg-badge {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}

.msg-num {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    display: block !important;
    line-height: 1 !important;
}

.msg-lbl {
    font-size: 0.62rem !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-top: 4px !important;
    display: block !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.25) !important;
    padding: 8px 16px !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}

/* ── HERO ── */
.hero-wrap {
    min-height: 92vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 40px;
    text-align: center;
    position: relative;
}

.hero-wrap::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse 40% 30% at 25% 30%, rgba(99,102,241,0.1) 0%, transparent 60%),
        radial-gradient(ellipse 30% 40% at 75% 70%, rgba(6,182,212,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 10%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 100px;
    padding: 8px 22px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 32px;
    position: relative;
    z-index: 1;
}

.hero-badge-dot {
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.hero-h1 {
    font-size: 5rem;
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 24px;
    position: relative;
    z-index: 1;
}

.hero-h1 span.grad {
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 35%, #06b6d4 65%, #ffffff 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: flow 6s linear infinite;
}

@keyframes flow {
    0% { background-position: 0% center; }
    100% { background-position: 300% center; }
}

.hero-p {
    font-size: 1.1rem;
    color: #64748b;
    max-width: 500px;
    line-height: 1.75;
    margin: 0 auto 48px;
    position: relative;
    z-index: 1;
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px;
    overflow: hidden;
    max-width: 640px;
    margin: 0 auto 64px;
    position: relative;
    z-index: 1;
}

.hero-stat {
    background: rgba(6,8,16,0.8);
    padding: 20px 16px;
    text-align: center;
}

.hs-val {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}

.hs-lbl {
    font-size: 0.62rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
    display: block;
}

.cap-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    max-width: 900px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

.cap-card {
    background: rgba(10,13,28,0.8);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: left;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.cap-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    opacity: 0;
    transition: opacity 0.25s ease;
}

.cap-card:hover {
    border-color: rgba(99,102,241,0.2);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.1);
}

.cap-card:hover::before { opacity: 1; }

.cap-ic { font-size: 1.6rem; margin-bottom: 10px; display: block; }
.cap-name { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }
.cap-desc { color: #475569; font-size: 0.73rem; line-height: 1.5; }

/* ── CHAT ── */
.stChatInputContainer {
    background: rgba(10,13,28,0.9) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
}

.stChatInputContainer:focus-within {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.06) !important;
}

/* ── CARDS ── */
.insight-card {
    background: rgba(10,13,28,0.6);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.insight-card:hover {
    border-color: rgba(99,102,241,0.15);
    background: rgba(99,102,241,0.03);
}

.ic-head {
    color: #a5b4fc;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.ic-body {
    color: #64748b;
    font-size: 0.8rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

.entity-tag {
    display: inline-block;
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.15);
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px;
    font-size: 0.74rem;
    color: #22d3ee;
    transition: all 0.15s ease;
}

.entity-tag:hover {
    background: rgba(6,182,212,0.13);
    transform: scale(1.03);
}

.evidence-item {
    background: rgba(99,102,241,0.04);
    border-left: 2px solid rgba(99,102,241,0.4);
    border-radius: 0 8px 8px 0;
    padding: 9px 13px;
    margin: 7px 0;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.55;
}

.quiz-card {
    background: rgba(10,13,28,0.6);
    border: 1px solid rgba(139,92,246,0.12);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.quiz-q {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 14px;
    line-height: 1.4;
}

.prompt-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 100px;
    padding: 7px 16px;
    font-size: 0.77rem;
    color: #a5b4fc;
    cursor: pointer;
    margin: 4px;
    transition: all 0.2s ease;
}

.prompt-chip:hover {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.25);
    color: #c7d2fe;
    transform: translateY(-1px);
}

.resource-yt {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.12);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.resource-yt:hover {
    border-color: rgba(239,68,68,0.25);
    background: rgba(239,68,68,0.08);
    transform: translateX(2px);
}

.resource-link {
    background: rgba(10,13,28,0.5);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.resource-link:hover {
    border-color: rgba(99,102,241,0.15);
    transform: translateX(2px);
}

div[data-testid="stExpander"] {
    background: rgba(10,13,28,0.4) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a5b4fc !important;
}

.stSelectbox > div > div {
    background: rgba(10,13,28,0.8) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 9px !important;
}

.stFileUploader {
    border: 2px dashed rgba(99,102,241,0.18) !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.02) !important;
}

.stFileUploader:hover {
    border-color: rgba(99,102,241,0.35) !important;
    background: rgba(99,102,241,0.05) !important;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }

hr { border-color: rgba(255,255,255,0.04) !important; }
</style>
"""


def logo():
    st.markdown("""
    <div class='dm-logo'>
        <div class='dm-logo-icon'>🧠</div>
        <div class='dm-logo-text'>
            <span class='dm-logo-name'>DocuMind AI</span>
            <span class='dm-logo-tag'>Document Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def gen_prompts(text: str) -> list:
    try:
        fn = lazy("utils.llm_provider", "get_shared_llm")
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate exactly 6 specific interesting questions a user would ask about this document.
Document: {text}
Rules: specific to content, mix factual/analytical/summary, under 10 words each, one per line, no numbering.
Questions:"""
        )
        llm = fn(temperature=0.5)
        result = (prompt | llm).invoke({"text": text[:1500]})
        lines = [l.strip() for l in result.content.strip().split("\n") if l.strip() and len(l.strip()) > 5]
        return lines[:6]
    except Exception:
        return ["What is this document about?", "Summarize the key points",
                "What are the main findings?", "What actions are recommended?",
                "What are the risks mentioned?", "Who is this document for?"]


def gen_quiz(text: str) -> list:
    try:
        fn = lazy("utils.llm_provider", "get_shared_llm")
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate exactly 5 multiple choice quiz questions from this document.
Document: {text}
For each use EXACTLY this format:
Q: [question]
A: [correct answer]
B: [wrong option]
C: [wrong option]
D: [wrong option]
ANSWER: A
Separate questions with blank lines."""
        )
        llm = fn(temperature=0.3)
        result = (prompt | llm).invoke({"text": text[:2500]})
        questions = []
        for block in result.content.strip().split("\n\n"):
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            q = {}
            opts = []
            for line in lines:
                if line.startswith("Q:"):
                    q["question"] = line[2:].strip()
                elif line.startswith(("A:", "B:", "C:", "D:")):
                    opts.append((line[0], line[2:].strip()))
                elif line.startswith("ANSWER:"):
                    q["answer"] = line[7:].strip()
            if q.get("question") and opts:
                q["options"] = opts
                questions.append(q)
        return questions[:5]
    except Exception:
        return []


def tts_html(text: str) -> str:
    clean = text[:600].replace('"', '').replace("'", "").replace("\n", " ")
    return f"""
    <script>
    var dmUtterance = null;
    function dmSpeak() {{
        window.speechSynthesis.cancel();
        dmUtterance = new SpeechSynthesisUtterance("{clean}");
        dmUtterance.rate = 0.92; dmUtterance.pitch = 1; dmUtterance.volume = 1;
        window.speechSynthesis.speak(dmUtterance);
    }}
    function dmStop() {{ window.speechSynthesis.cancel(); }}
    </script>
    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
        <button onclick="dmSpeak()" style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:8px;color:#a5b4fc;padding:5px 14px;font-size:0.76rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;">🔊 Read Aloud</button>
        <button onclick="dmStop()" style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.15);border-radius:8px;color:#fca5a5;padding:5px 14px;font-size:0.76rem;cursor:pointer;font-family:Inter,sans-serif;font-weight:600;">⏹ Stop</button>
    </div>"""


def gen_file(content: str, fmt: str, topic: str = "document") -> bytes:
    if fmt == "txt":
        return content.encode("utf-8")

    elif fmt == "docx":
        from docx import Document
        doc = Document()
        doc.add_heading("DocuMind AI — Generated Report", 0)
        for line in content.split("\n"):
            if line.strip():
                if line.startswith("##"):
                    doc.add_heading(line.replace("##", "").strip(), 2)
                elif line.startswith("#"):
                    doc.add_heading(line.replace("#", "").strip(), 1)
                else:
                    doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    elif fmt == "csv":
        import pandas as pd
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        df = pd.DataFrame({"Content": lines})
        return df.to_csv(index=False).encode("utf-8")

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
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 6))
            doc.build(story)
            return buf.getvalue()
        except Exception:
            return content.encode("utf-8")

    return content.encode("utf-8")


def gen_chart(text: str, question: str):
    try:
        import pandas as pd
        import plotly.express as px
        fn = lazy("utils.llm_provider", "get_shared_llm")
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["text", "question"],
            template="""Extract numerical data from this document to create a chart for: {question}

Document: {text}

Return ONLY valid Python dict like this:
{{"labels": ["A","B","C"], "values": [10, 20, 30], "title": "Chart Title", "type": "bar"}}
type can be: bar, pie, line
Return ONLY the dict. No explanation."""
        )
        llm = fn(temperature=0)
        result = (prompt | llm).invoke({"text": text[:2000], "question": question})
        content = result.content.strip()
        content = content.replace("```python", "").replace("```json", "").replace("```", "").strip()
        data = eval(content)
        df = pd.DataFrame({"Label": data["labels"], "Value": data["values"]})
        chart_type = data.get("type", "bar")
        title = data.get("title", "Chart")
        if chart_type == "pie":
            fig = px.pie(df, names="Label", values="Value", title=title,
                        color_discrete_sequence=px.colors.sequential.Plasma)
        elif chart_type == "line":
            fig = px.line(df, x="Label", y="Value", title=title,
                         color_discrete_sequence=["#6366f1"])
        else:
            fig = px.bar(df, x="Label", y="Value", title=title,
                        color_discrete_sequence=["#6366f1"])
        fig.update_layout(
            paper_bgcolor="rgba(6,8,16,0)",
            plot_bgcolor="rgba(6,8,16,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0"
        )
        return fig
    except Exception:
        return None


def process_doc(uploaded_file, deps) -> bool:
    import tempfile
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        is_valid, msg = deps["validate_file"](tmp_path)
        if not is_valid:
            st.error(msg)
            return False

        st.session_state.file_path = tmp_path
        st.session_state.doc_name = uploaded_file.name

        pb = st.progress(0, "Reading document...")
        text = deps["load_document"](tmp_path)
        st.session_state.doc_text = text

        pb.progress(30, "Chunking...")
        chunks = deps["chunk_text"](text)

        pb.progress(55, "Building vector index...")
        deps["create_vector_store"](chunks)

        pb.progress(75, "Building keyword index...")
        deps["create_bm25_index"](chunks)

        pb.progress(90, "Generating smart prompts...")
        st.session_state.prompts = gen_prompts(text)

        pb.progress(100, "✅ Ready!")
        time.sleep(0.2)
        pb.empty()

        st.session_state.doc_ready = True
        st.session_state.ins_loaded = False
        st.session_state.ent_loaded = False
        st.session_state.map_loaded = False
        st.session_state.quiz_loaded = False
        st.session_state.entities = {}
        st.session_state.insights = {}
        st.session_state.doc_map = []
        st.session_state.quiz = []
        st.session_state.chat = []
        st.session_state.memory = st.session_state.fmm.get_memory(uploaded_file.name)

        existing = [d["name"] for d in st.session_state.multi_docs]
        if uploaded_file.name not in existing:
            st.session_state.multi_docs.append({"name": uploaded_file.name, "path": tmp_path, "text": text})

        return True
    except Exception as e:
        st.error(deps["handle_error"](e, "upload"))
        return False


def show_msg(role: str, content: str, show_tts: bool = False):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)
            if show_tts:
                components.html(tts_html(content), height=50)


def answer_q(question: str, deps):
    is_valid, msg = deps["validate_question"](question)
    if not is_valid:
        st.warning(msg)
        return

    show_msg("human", question)
    st.session_state.chat.append({"role": "human", "content": question})

    with st.spinner("🧠 Thinking..."):
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

    show_msg("assistant", answer, show_tts=True)

    if evidence:
        with st.expander("📎 Evidence Sources"):
            for i, chunk in enumerate(evidence, 1):
                st.markdown(f"<div class='evidence-item'><strong style='color:#a5b4fc;'>Source {i}</strong><br>{chunk}</div>", unsafe_allow_html=True)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.last_q = question
    st.session_state.last_a = answer

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("💡 Suggestions", key=f"s{len(st.session_state.chat)}", use_container_width=True):
            fn = lazy("agents.suggestion_agent", "generate_suggestions")
            with st.spinner(""):
                sugs = fn(question, answer)
            for s in sugs:
                st.markdown(f"• {s}")

    with c2:
        if st.button("🌐 Resources", key=f"r{len(st.session_state.chat)}", use_container_width=True):
            fn = lazy("agents.knowledge_agent", "expand_knowledge")
            with st.spinner("Finding..."):
                k = fn(question, answer)
            if k:
                if k.get("YOUTUBE_SEARCHES"):
                    for s in k["YOUTUBE_SEARCHES"]:
                        url = f"https://www.youtube.com/results?search_query={s.replace(' ', '+')}"
                        st.markdown(f"<div class='resource-yt'><a href='{url}' target='_blank' style='color:#fca5a5;text-decoration:none;'>▶ {s}</a></div>", unsafe_allow_html=True)
                if k.get("RELATED_TOPICS"):
                    for t in k["RELATED_TOPICS"]:
                        url = f"https://www.google.com/search?q={t.replace(' ', '+')}"
                        st.markdown(f"<div class='resource-link'><a href='{url}' target='_blank' style='color:#a5b4fc;text-decoration:none;'>🔍 {t}</a></div>", unsafe_allow_html=True)

    with c3:
        if st.button("📊 Chart", key=f"ch{len(st.session_state.chat)}", use_container_width=True):
            with st.spinner("Generating chart..."):
                fig = gen_chart(st.session_state.doc_text[:2000], question)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not generate chart for this question.")

    with c4:
        fmt = st.selectbox("Format", ["txt", "docx", "pdf", "csv"], key=f"fmt{len(st.session_state.chat)}", label_visibility="collapsed")
        file_bytes = gen_file(answer, fmt)
        st.download_button(
            "📥 Export",
            data=file_bytes,
            file_name=f"answer.{fmt}",
            mime="application/octet-stream",
            use_container_width=True,
            key=f"dl{len(st.session_state.chat)}"
        )


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    deps = load_core()
    init(deps)

    with st.sidebar:
        logo()

        st.markdown("<span class='sb-label'>Upload Document</span>", unsafe_allow_html=True)
        f = st.file_uploader("f", type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"], label_visibility="collapsed")

        if f:
            kb = f.size / 1024
            st.markdown(f"<div class='file-card'><div class='file-name'>📄 {f.name}</div><div class='file-size'>{kb:.0f} KB</div></div>", unsafe_allow_html=True)
            if st.button("🚀 Process Document", use_container_width=True):
                if process_doc(f, deps):
                    st.success("✅ Ready!")
                    st.rerun()

        if st.session_state.doc_ready:
            st.markdown("---")
            st.markdown("<span class='sb-label'>Answer Style</span>", unsafe_allow_html=True)
            mode = st.selectbox("m", ["detailed", "quick", "bullet", "beginner", "executive", "table"],
                format_func=lambda x: {"detailed": "📝 Detailed", "quick": "⚡ Quick", "bullet": "• Bullets",
                "beginner": "🎓 Beginner", "executive": "💼 Executive", "table": "📊 Table"}[x],
                label_visibility="collapsed")
            st.session_state.answer_mode = mode

            st.markdown("---")
            st.markdown("<span class='sb-label'>Active Document</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='file-card'><div class='file-name'>📄 {st.session_state.doc_name}</div></div>", unsafe_allow_html=True)

            st.markdown("<span class='sb-label'>Session</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='msg-badge'><span class='msg-num'>{len(st.session_state.chat)}</span><span class='msg-lbl'>Messages</span></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<span class='sb-label'>Quick Actions</span>", unsafe_allow_html=True)
            for action in ["📝 Summarize document", "✅ Extract action items",
                          "⚠️ Identify all risks", "📊 Extract key metrics", "❓ Generate FAQ"]:
                if st.button(action, use_container_width=True, key=f"qa{action[:6]}"):
                    with st.spinner("Working..."):
                        fn = lazy("agents.document_action_agent", "perform_document_action")
                        r = fn(action=action, context=st.session_state.doc_text[:4000], file_name=st.session_state.doc_name)
                    if r["success"]:
                        st.session_state.chat.append({"role": "assistant", "content": f"**{action}**\n\n{r['result']}"})
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
                    for k in ["doc_ready", "ins_loaded", "ent_loaded", "map_loaded", "quiz_loaded"]:
                        st.session_state[k] = False
                    for k in ["file_path", "doc_name", "doc_text"]:
                        st.session_state[k] = ""
                    st.session_state.update({"entities": {}, "insights": {}, "doc_map": [], "chat": [], "prompts": []})
                    st.session_state.memory = deps["SessionMemory"]()
                    st.rerun()

    if not st.session_state.doc_ready:
        st.markdown("""
        <div class='hero-wrap'>
            <div class='hero-badge'><div class='hero-badge-dot'></div>Live — AI-Powered</div>
            <div class='hero-h1'><span class='grad'>DocuMind AI</span></div>
            <div class='hero-p'>Upload any document. Ask anything in plain English. Get instant expert-level answers, charts, quizzes, and knowledge — all private.</div>
            <div class='hero-grid'>
                <div class='hero-stat'><span class='hs-val'>6+</span><span class='hs-lbl'>Formats</span></div>
                <div class='hero-stat'><span class='hs-val'>5</span><span class='hs-lbl'>AI Agents</span></div>
                <div class='hero-stat'><span class='hs-val'>20+</span><span class='hs-lbl'>Features</span></div>
                <div class='hero-stat'><span class='hs-val'>100%</span><span class='hs-lbl'>Private</span></div>
            </div>
            <div class='cap-grid'>
                <div class='cap-card'><span class='cap-ic'>⚡</span><div class='cap-name'>Ultra-Fast</div><div class='cap-desc'>Groq-powered AI — responses in 1-2 seconds</div></div>
                <div class='cap-card'><span class='cap-ic'>📊</span><div class='cap-name'>Auto Charts</div><div class='cap-desc'>Generate visual charts from document data</div></div>
                <div class='cap-card'><span class='cap-ic'>🧩</span><div class='cap-name'>Quiz Mode</div><div class='cap-desc'>Auto-generated MCQ quiz from document</div></div>
                <div class='cap-card'><span class='cap-ic'>🔊</span><div class='cap-name'>Read Aloud</div><div class='cap-desc'>Browser TTS for every answer</div></div>
                <div class='cap-card'><span class='cap-ic'>🌐</span><div class='cap-name'>Web Resources</div><div class='cap-desc'>Real YouTube and Google links</div></div>
                <div class='cap-card'><span class='cap-ic'>💡</span><div class='cap-name'>Smart Prompts</div><div class='cap-desc'>Auto-generated questions from your document</div></div>
                <div class='cap-card'><span class='cap-ic'>📥</span><div class='cap-name'>Export Anything</div><div class='cap-desc'>Download as PDF, DOCX, CSV, or TXT</div></div>
                <div class='cap-card'><span class='cap-ic'>🔒</span><div class='cap-name'>100% Private</div><div class='cap-desc'>Nothing leaves your machine</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    tabs = st.tabs(["💬 Chat", "💡 Smart Prompts", "🧩 Quiz", "🔍 Insights", "📊 Entities", "🗺️ Map", "✂️ Selection"])

    with tabs[0]:
        if len(st.session_state.multi_docs) > 1:
            with st.expander("🗂️ Multi-Document Query"):
                mq = st.text_input("Ask across all documents:", placeholder="Compare topics...")
                if st.button("🔍 Query All Documents"):
                    if mq:
                        with st.spinner("Querying all documents..."):
                            fn = lazy("agents.multi_document_agent", "query_multiple_documents")
                            ans = fn(mq, st.session_state.multi_docs)
                        show_msg("assistant", ans, show_tts=True)
                        st.session_state.chat.append({"role": "assistant", "content": ans})

        for msg in st.session_state.chat:
            show_msg(msg["role"], msg["content"])

        ci, cv = st.columns([6, 1])
        with ci:
            question = st.chat_input("Ask anything about your document...")
        with cv:
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
                if st.button("🎤", help="Voice on HTTPS deployment"):
                    st.info("🎤 Voice works after Streamlit Cloud deployment")

        if question:
            answer_q(question, deps)

    with tabs[1]:
        st.markdown("### 💡 Smart Prompts")
        st.caption("Auto-generated from your document — click any to ask")
        prompts = st.session_state.prompts
        if prompts:
            cols = st.columns(2)
            for i, p in enumerate(prompts):
                with cols[i % 2]:
                    if st.button(f"💬 {p}", key=f"pr{i}", use_container_width=True):
                        answer_q(p, deps)
        else:
            if st.button("💡 Generate Prompts", use_container_width=True):
                with st.spinner("Analyzing..."):
                    st.session_state.prompts = gen_prompts(st.session_state.doc_text)
                st.rerun()

    with tabs[2]:
        st.markdown("### 🧩 Document Quiz")
        if not st.session_state.quiz_loaded:
            st.markdown("<div class='insight-card'><div class='ic-body'>Test your understanding with an auto-generated quiz from your document.</div></div>", unsafe_allow_html=True)
            if st.button("🧩 Generate Quiz", use_container_width=True):
                with st.spinner("Creating quiz..."):
                    st.session_state.quiz = gen_quiz(st.session_state.doc_text)
                    st.session_state.quiz_loaded = True
                st.rerun()
        else:
            if not st.session_state.quiz:
                st.warning("Could not generate quiz. Try again.")
                if st.button("Retry"):
                    st.session_state.quiz_loaded = False
                    st.rerun()
            else:
                score_key = "quiz_score"
                if score_key not in st.session_state:
                    st.session_state[score_key] = {}

                for i, q in enumerate(st.session_state.quiz, 1):
                    st.markdown(f"<div class='quiz-card'><div class='quiz-q'>Q{i}. {q.get('question', '')}</div></div>", unsafe_allow_html=True)
                    if q.get("options"):
                        opts = [f"{o[0]}. {o[1]}" for o in q["options"]]
                        user = st.radio(f"Q{i}", opts, key=f"qr{i}", label_visibility="collapsed")
                        if st.button(f"✓ Check Q{i}", key=f"qc{i}"):
                            correct = q.get("answer", "A")
                            if user and user.startswith(correct):
                                st.success("✅ Correct!")
                                st.session_state[score_key][i] = True
                            else:
                                correct_text = next((f"{o[0]}. {o[1]}" for o in q["options"] if o[0] == correct), correct)
                                st.error(f"❌ Correct: {correct_text}")
                                st.session_state[score_key][i] = False

                answered = len(st.session_state[score_key])
                if answered > 0:
                    correct_count = sum(1 for v in st.session_state[score_key].values() if v)
                    st.markdown(f"**Score: {correct_count}/{answered}**")

    with tabs[3]:
        st.markdown("### 🔍 Living Insight Layer")
        if not st.session_state.ins_loaded:
            st.markdown("<div class='insight-card'><div class='ic-body'>Generate AI insights: summary, findings, risks, action items, next steps.</div></div>", unsafe_allow_html=True)
            if st.button("🧠 Generate Insights", use_container_width=True):
                with st.spinner("Analyzing..."):
                    fn = lazy("agents.insight_agent", "generate_insights")
                    st.session_state.insights = fn(st.session_state.doc_text[:4000])
                    st.session_state.ins_loaded = True
                st.rerun()
        else:
            im = {"SUMMARY": ("📄", "Summary"), "KEY_FINDINGS": ("🎯", "Key Findings"),
                  "ACTION_ITEMS": ("✅", "Action Items"), "RISKS": ("⚠️", "Risks"),
                  "DECISIONS": ("💡", "Decisions"), "NEXT_STEPS": ("🚀", "Next Steps")}
            for key, (icon, title) in im.items():
                if st.session_state.insights.get(key):
                    st.markdown(f"<div class='insight-card'><div class='ic-head'>{icon} {title}</div><div class='ic-body'>{st.session_state.insights[key]}</div></div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("### 📊 Entity Dashboard")
        if not st.session_state.ent_loaded:
            st.markdown("<div class='insight-card'><div class='ic-body'>Extract people, organizations, dates, locations, and key terms.</div></div>", unsafe_allow_html=True)
            if st.button("🔍 Extract Entities", use_container_width=True):
                with st.spinner("Extracting..."):
                    fn = lazy("agents.entity_agent", "extract_entities")
                    st.session_state.entities = fn(st.session_state.doc_text[:3000])
                    st.session_state.ent_loaded = True
                st.rerun()
        else:
            icons = {"PEOPLE": "👤", "ORGANIZATIONS": "🏢", "DATES": "📅", "LOCATIONS": "📍", "KEY_TERMS": "🔑", "ACTION_ITEMS": "✅"}
            for et, vals in st.session_state.entities.items():
                if vals:
                    ic = icons.get(et, "•")
                    st.markdown(f"**{ic} {et}**")
                    st.markdown("".join([f"<span class='entity-tag'>{v}</span>" for v in vals]), unsafe_allow_html=True)
                    st.markdown("")

    with tabs[5]:
        st.markdown("### 🗺️ Document Map")
        if not st.session_state.map_loaded:
            st.markdown("<div class='insight-card'><div class='ic-body'>Build a smart navigation map of document sections.</div></div>", unsafe_allow_html=True)
            if st.button("🗺️ Build Map", use_container_width=True):
                with st.spinner("Mapping..."):
                    fn = lazy("agents.document_map_agent", "extract_document_map")
                    st.session_state.doc_map = fn(st.session_state.doc_text[:5000])
                    st.session_state.map_loaded = True
                st.rerun()
        else:
            for section in st.session_state.doc_map:
                with st.expander(f"📍 {section['title']}"):
                    st.markdown(section.get("description", ""))
                    if st.button("Ask about this", key=f"m{section['title'][:12]}"):
                        answer_q(f"Tell me about: {section['title']}", deps)

    with tabs[6]:
        st.markdown("### ✂️ Ask by Selection")
        st.markdown("<div class='insight-card'><div class='ic-body'>Paste any specific text from your document and analyze it precisely.</div></div>", unsafe_allow_html=True)
        sel = st.text_area("Paste text:", placeholder="Any paragraph, table, or section...", height=120)
        act = st.selectbox("Action:", ["Explain this", "Simplify this", "Find issues", "Summarize", "Key points", "Rewrite professionally"])

        if st.button("🔍 Analyze Selection", use_container_width=True):
            if sel.strip():
                with st.spinner("Analyzing..."):
                    fn = lazy("agents.selection_agent", "ask_about_selection")
                    res = fn(sel, act)
                st.markdown(f"<div class='insight-card'><div class='ic-head'>✨ Result</div><div class='ic-body'>{res}</div></div>", unsafe_allow_html=True)
                components.html(tts_html(res), height=50)
                fmt2 = st.selectbox("Export as:", ["txt", "docx", "pdf"], key="sel_fmt")
                fb = gen_file(res, fmt2)
                st.download_button("📥 Download", data=fb, file_name=f"selection.{fmt2}", mime="application/octet-stream")
            else:
                st.warning("Paste some text first.")


if __name__ == "__main__":
    main()