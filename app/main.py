import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from loguru import logger


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def load_core_dependencies():
    from ingestion.document_loader import load_document
    from chunking.text_chunker import chunk_text
    from vector_store.faiss_store import create_vector_store
    from vector_store.bm25_store import create_bm25_index
    from graph.workflow import run_workflow
    from memory.session_memory import SessionMemory
    from memory.file_memory_manager import FileMemoryManager
    from utils.validator import validate_file, validate_question
    from utils.error_handler import handle_error
    from utils.performance import PerformanceTracker
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
        "PerformanceTracker": PerformanceTracker
    }


def get_agent(name: str):
    agents = {
        "suggestions": ("agents.suggestion_agent", "generate_suggestions"),
        "knowledge": ("agents.knowledge_agent", "expand_knowledge"),
        "entity": ("agents.entity_agent", "extract_entities"),
        "insight": ("agents.insight_agent", "generate_insights"),
        "document_map": ("agents.document_map_agent", "extract_document_map"),
        "selection": ("agents.selection_agent", "ask_about_selection"),
        "action": ("agents.document_action_agent", "perform_document_action"),
        "multi_doc": ("agents.multi_document_agent", "query_multiple_documents"),
        "voice": ("agents.voice_agent", "transcribe_audio_file"),
    }
    if name in agents:
        module_path, func_name = agents[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    return None


def get_export():
    from tools.file_export_tool import export_as_txt, export_as_docx
    return export_as_txt, export_as_docx


def initialize_session(deps: dict):
    defaults = {
        "memory": deps["SessionMemory"](),
        "file_memory_manager": deps["FileMemoryManager"](),
        "chat_history": [],
        "file_path": "",
        "document_processed": False,
        "document_name": "",
        "document_text": "",
        "entities": {},
        "insights": {},
        "document_map": [],
        "answer_mode": "detailed",
        "awaiting_clarification": False,
        "pending_question": None,
        "clarification_questions": [],
        "multi_documents": [],
        "insights_loaded": False,
        "entities_loaded": False,
        "map_loaded": False,
        "show_voice_tip": False,
        "last_question": "",
        "last_answer": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #050508;
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(88,166,255,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(63,185,80,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
        color: #e6edf3;
        min-height: 100vh;
    }

    section[data-testid="stSidebar"] {
        background: rgba(13,17,23,0.98) !important;
        border-right: 1px solid rgba(88,166,255,0.08) !important;
        backdrop-filter: blur(20px);
    }

    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    .hero-section {
        text-align: center;
        padding: 48px 24px 32px;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 50% 0%, rgba(88,166,255,0.12) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.2);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #58a6ff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #ffffff 0%, #58a6ff 50%, #3fb950 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 32px;
        line-height: 1.6;
    }

    .stats-row {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 24px;
        flex-wrap: wrap;
    }

    .stat-item {
        text-align: center;
    }

    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        display: block;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 40px;
        padding: 0 20px;
    }

    .feature-card {
        background: rgba(22,27,34,0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(88,166,255,0.3), transparent);
    }

    .feature-card:hover {
        border-color: rgba(88,166,255,0.2);
        background: rgba(88,166,255,0.05);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(88,166,255,0.1);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
        display: block;
    }

    .feature-title {
        color: #e6edf3;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .feature-desc {
        color: #8b949e;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    .sidebar-section {
        margin-bottom: 20px;
    }

    .sidebar-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: rgba(139,148,158,0.8) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        display: block;
    }

    .file-pill {
        background: rgba(63,185,80,0.08);
        border: 1px solid rgba(63,185,80,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .file-pill-name {
        color: #3fb950;
        font-size: 0.83rem;
        font-weight: 500;
    }

    .file-pill-size {
        color: #8b949e;
        font-size: 0.75rem;
        margin-left: auto;
    }

    .msg-count {
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.12);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .msg-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        display: block;
    }

    .msg-label {
        font-size: 0.68rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(31,111,235,0.3) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(88,166,255,0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .action-btn > button {
        background: rgba(88,166,255,0.08) !important;
        border: 1px solid rgba(88,166,255,0.15) !important;
        color: #58a6ff !important;
        box-shadow: none !important;
    }

    .action-btn > button:hover {
        background: rgba(88,166,255,0.15) !important;
        border-color: rgba(88,166,255,0.3) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 4px 0 !important;
    }

    .stChatInputContainer {
        background: rgba(22,27,34,0.8) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stChatInputContainer:focus-within {
        border-color: rgba(88,166,255,0.4) !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.08) !important;
    }

    .evidence-container {
        background: rgba(88,166,255,0.03);
        border: 1px solid rgba(88,166,255,0.1);
        border-radius: 12px;
        padding: 4px;
        margin-top: 8px;
    }

    .evidence-chunk {
        background: rgba(13,17,23,0.6);
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px;
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.5;
    }

    .insight-card {
        background: rgba(63,185,80,0.04);
        border: 1px solid rgba(63,185,80,0.12);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }

    .insight-card:hover {
        border-color: rgba(63,185,80,0.25);
        background: rgba(63,185,80,0.07);
    }

    .insight-icon-title {
        color: #3fb950;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .insight-content {
        color: #8b949e;
        font-size: 0.83rem;
        line-height: 1.6;
    }

    .entity-tag {
        display: inline-block;
        background: rgba(210,153,34,0.08);
        border: 1px solid rgba(210,153,34,0.2);
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.78rem;
        color: #d29922;
        transition: all 0.2s ease;
    }

    .entity-tag:hover {
        background: rgba(210,153,34,0.15);
    }

    .resource-card {
        background: rgba(22,27,34,0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .resource-card:hover {
        border-color: rgba(88,166,255,0.2);
        background: rgba(88,166,255,0.04);
        transform: translateX(3px);
    }

    .resource-card a {
        color: #58a6ff !important;
        text-decoration: none !important;
        font-weight: 500;
        font-size: 0.88rem;
    }

    .resource-card a:hover {
        text-decoration: underline !important;
    }

    .resource-meta {
        color: #8b949e;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    .youtube-card {
        background: rgba(255,0,0,0.05);
        border: 1px solid rgba(255,0,0,0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }

    .youtube-card:hover {
        border-color: rgba(255,0,0,0.3);
        background: rgba(255,0,0,0.08);
    }

    .youtube-card a {
        color: #ff6b6b !important;
        text-decoration: none !important;
        font-weight: 500;
        font-size: 0.88rem;
    }

    .processing-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        font-size: 0.85rem;
        color: #8b949e;
    }

    .processing-step.active {
        color: #58a6ff;
    }

    .processing-step.done {
        color: #3fb950;
    }

    .tab-content {
        padding: 20px 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(22,27,34,0.4) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
    }

    .stSelectbox > div > div {
        background: rgba(22,27,34,0.8) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    .stFileUploader {
        border: 2px dashed rgba(88,166,255,0.2) !important;
        border-radius: 12px !important;
        background: rgba(88,166,255,0.02) !important;
        transition: all 0.2s ease !important;
    }

    .stFileUploader:hover {
        border-color: rgba(88,166,255,0.4) !important;
        background: rgba(88,166,255,0.05) !important;
    }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(88,166,255,0.2); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(88,166,255,0.4); }

    hr { border-color: rgba(255,255,255,0.06) !important; }

    .stSuccess {
        background: rgba(63,185,80,0.08) !important;
        border: 1px solid rgba(63,185,80,0.2) !important;
        border-radius: 10px !important;
    }

    .stError {
        background: rgba(248,81,73,0.08) !important;
        border: 1px solid rgba(248,81,73,0.2) !important;
        border-radius: 10px !important;
    }

    .stWarning {
        background: rgba(210,153,34,0.08) !important;
        border: 1px solid rgba(210,153,34,0.2) !important;
        border-radius: 10px !important;
    }

    .stInfo {
        background: rgba(88,166,255,0.06) !important;
        border: 1px solid rgba(88,166,255,0.15) !important;
        border-radius: 10px !important;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #58a6ff;
        animation: pulse 1.5s ease-in-out infinite;
        margin: 0 2px;
    }

    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """, unsafe_allow_html=True)


def show_hero():
    st.markdown("""
    <div class='hero-section'>
        <div class='hero-badge'>✦ AI-Powered Document Intelligence</div>
        <div class='hero-title'>DocuMind AI</div>
        <div class='hero-subtitle'>
            Upload any document and unlock instant intelligence — powered by
            Google Gemini, LangGraph agents, and hybrid search technology
        </div>
        <div class='stats-row'>
            <div class='stat-item'>
                <span class='stat-number'>6+</span>
                <span class='stat-label'>File Formats</span>
            </div>
            <div class='stat-item'>
                <span class='stat-number'>5</span>
                <span class='stat-label'>AI Agents</span>
            </div>
            <div class='stat-item'>
                <span class='stat-number'>20+</span>
                <span class='stat-label'>Features</span>
            </div>
            <div class='stat-item'>
                <span class='stat-number'>100%</span>
                <span class='stat-label'>Private</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_welcome_features():
    st.markdown("""
    <div class='feature-grid'>
        <div class='feature-card'>
            <span class='feature-icon'>🔍</span>
            <div class='feature-title'>Hybrid Search</div>
            <div class='feature-desc'>FAISS semantic + BM25 keyword with Reciprocal Rank Fusion</div>
        </div>
        <div class='feature-card'>
            <span class='feature-icon'>🤖</span>
            <div class='feature-title'>Multi-Agent AI</div>
            <div class='feature-desc'>Intent, Planner, Retriever, QA agents orchestrated by LangGraph</div>
        </div>
        <div class='feature-card'>
            <span class='feature-icon'>📊</span>
            <div class='feature-title'>Data Computation</div>
            <div class='feature-desc'>Actual calculations on Excel and CSV with Pandas DataFrame agent</div>
        </div>
        <div class='feature-card'>
            <span class='feature-icon'>💬</span>
            <div class='feature-title'>Session Memory</div>
            <div class='feature-desc'>Per-document conversation memory with full multi-turn support</div>
        </div>
        <div class='feature-card'>
            <span class='feature-icon'>🌐</span>
            <div class='feature-title'>Knowledge Expansion</div>
            <div class='feature-desc'>Real YouTube links and web resources related to your document</div>
        </div>
        <div class='feature-card'>
            <span class='feature-icon'>🔒</span>
            <div class='feature-title'>100% Private</div>
            <div class='feature-desc'>Documents never leave your machine. Zero data exposure.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_message(role: str, content: str):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)


def show_evidence(evidence: list):
    if not evidence:
        return
    with st.expander("📎 Evidence Sources Used"):
        st.markdown("<div class='evidence-container'>", unsafe_allow_html=True)
        for i, chunk in enumerate(evidence, 1):
            st.markdown(
                f"<div class='evidence-chunk'><strong style='color:#58a6ff;'>Source {i}</strong><br>{chunk}</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)


def show_resources(question: str, answer: str):
    fn = get_agent("knowledge")
    if not fn:
        return

    with st.spinner("🌐 Finding related resources..."):
        knowledge = fn(question, answer)

    if not knowledge:
        return

    st.markdown("---")
    st.markdown("### 🌐 Related External Resources")

    col1, col2 = st.columns(2)

    with col1:
        if knowledge.get("YOUTUBE_SEARCHES"):
            st.markdown("#### 📺 YouTube — Search These")
            for search in knowledge["YOUTUBE_SEARCHES"]:
                query = search.strip().replace(" ", "+")
                url = f"https://www.youtube.com/results?search_query={query}"
                st.markdown(
                    f"""
                    <div class='youtube-card'>
                        <a href='{url}' target='_blank'>▶ {search}</a>
                        <div class='resource-meta'>Click to search on YouTube →</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if knowledge.get("RELATED_TOPICS"):
            st.markdown("#### 🔗 Related Topics")
            for topic in knowledge["RELATED_TOPICS"]:
                query = topic.strip().replace(" ", "+")
                wiki_url = f"https://en.wikipedia.org/wiki/Special:Search?search={query}"
                google_url = f"https://www.google.com/search?q={query}"
                st.markdown(
                    f"""
                    <div class='resource-card'>
                        <a href='{wiki_url}' target='_blank'>📖 {topic}</a>
                        <div class='resource-meta'>
                            <a href='{wiki_url}' target='_blank' style='color:#8b949e;'>Wikipedia</a>
                            &nbsp;·&nbsp;
                            <a href='{google_url}' target='_blank' style='color:#8b949e;'>Google</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col2:
        if knowledge.get("SIMILAR_RESOURCES"):
            st.markdown("#### 📚 Similar Resources")
            for resource in knowledge["SIMILAR_RESOURCES"]:
                query = resource.strip().replace(" ", "+")
                scholar_url = f"https://scholar.google.com/scholar?q={query}"
                st.markdown(
                    f"""
                    <div class='resource-card'>
                        <a href='{scholar_url}' target='_blank'>🎓 {resource}</a>
                        <div class='resource-meta'>Search on Google Scholar →</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if knowledge.get("LEARN_MORE"):
            st.markdown("#### 🎯 Recommended Learning")
            for item in knowledge["LEARN_MORE"]:
                query = item.strip().replace(" ", "+")
                url = f"https://www.google.com/search?q={query}"
                st.markdown(
                    f"""
                    <div class='resource-card'>
                        <a href='{url}' target='_blank'>💡 {item}</a>
                        <div class='resource-meta'>Search online →</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def process_document(uploaded_file, deps: dict) -> bool:
    import tempfile
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        is_valid, message = deps["validate_file"](tmp_path)
        if not is_valid:
            st.error(message)
            return False

        st.session_state.file_path = tmp_path
        st.session_state.document_name = uploaded_file.name

        progress = st.progress(0, text="Starting...")

        progress.progress(10, text="📖 Reading document...")
        text = deps["load_document"](tmp_path)
        st.session_state.document_text = text

        progress.progress(35, text="✂️ Chunking document...")
        chunks = deps["chunk_text"](text)

        progress.progress(60, text="🔢 Building vector index...")
        deps["create_vector_store"](chunks)

        progress.progress(80, text="🔍 Building keyword index...")
        deps["create_bm25_index"](chunks)

        progress.progress(100, text="✅ Complete!")

        import time
        time.sleep(0.5)
        progress.empty()

        st.session_state.document_processed = True
        st.session_state.insights_loaded = False
        st.session_state.entities_loaded = False
        st.session_state.map_loaded = False
        st.session_state.entities = {}
        st.session_state.insights = {}
        st.session_state.document_map = []
        st.session_state.memory = (
            st.session_state.file_memory_manager.get_memory(
                uploaded_file.name
            )
        )
        st.session_state.chat_history = []

        existing = [d["name"] for d in st.session_state.multi_documents]
        if uploaded_file.name not in existing:
            st.session_state.multi_documents.append({
                "name": uploaded_file.name,
                "path": tmp_path,
                "text": text
            })

        logger.info(f"Document processed: {uploaded_file.name}")
        return True

    except Exception as e:
        error_msg = deps["handle_error"](e, "document_processing")
        st.error(error_msg)
        return False


def handle_question(question: str, deps: dict):
    answer_mode = st.session_state.answer_mode

    is_valid, msg = deps["validate_question"](question)
    if not is_valid:
        st.warning(msg)
        return

    show_message("human", question)
    st.session_state.chat_history.append({
        "role": "human",
        "content": question
    })

    with st.spinner("🧠 Analysing your question..."):
        try:
            result = deps["run_workflow"](
                question=question,
                memory=st.session_state.memory,
                file_path=st.session_state.file_path,
                answer_mode=answer_mode
            )

            answer = result["answer"]
            evidence = result["evidence"]

            show_message("assistant", answer)
            show_evidence(evidence)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            st.session_state.last_question = question
            st.session_state.last_answer = answer

        except Exception as e:
            error_msg = deps["handle_error"](e, "workflow")
            st.error(error_msg)
            return

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "💡 Get Suggestions",
            key=f"sug_{len(st.session_state.chat_history)}",
            use_container_width=True
        ):
            fn = get_agent("suggestions")
            with st.spinner("Generating..."):
                suggestions = fn(question, answer)
            if suggestions:
                st.markdown("**Follow-up questions:**")
                for s in suggestions:
                    st.markdown(f"• {s}")

    with col2:
        if st.button(
            "🌐 Related Resources",
            key=f"res_{len(st.session_state.chat_history)}",
            use_container_width=True
        ):
            show_resources(question, answer)

    with col3:
        export_as_txt, export_as_docx = get_export()
        txt_path = export_as_txt(
            answer,
            f"answer_{len(st.session_state.chat_history)}"
        )
        with open(txt_path, "rb") as f:
            st.download_button(
                "📥 Download Answer",
                data=f,
                file_name="documind_answer.txt",
                use_container_width=True,
                key=f"dl_{len(st.session_state.chat_history)}"
            )


def main():
    apply_css()

    with st.spinner("⚡ Loading DocuMind AI..."):
        deps = load_core_dependencies()

    initialize_session(deps)

    with st.sidebar:
        st.markdown(
            "<span class='sidebar-label'>Document Upload</span>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            kb = uploaded_file.size / 1024
            st.markdown(
                f"""
                <div class='file-pill'>
                    <span>📄</span>
                    <span class='file-pill-name'>{uploaded_file.name}</span>
                    <span class='file-pill-size'>{kb:.0f} KB</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🚀 Process Document", use_container_width=True):
                success = process_document(uploaded_file, deps)
                if success:
                    st.success("✅ Ready to answer questions!")
                    st.rerun()

        if st.session_state.document_processed:
            st.markdown("---")
            st.markdown(
                "<span class='sidebar-label'>Answer Style</span>",
                unsafe_allow_html=True
            )
            mode = st.selectbox(
                "mode",
                options=["detailed", "quick", "bullet", "beginner", "executive", "table"],
                format_func=lambda x: {
                    "detailed": "📝 Detailed",
                    "quick": "⚡ Quick",
                    "bullet": "• Bullets",
                    "beginner": "🎓 Beginner",
                    "executive": "💼 Executive",
                    "table": "📊 Table"
                }[x],
                label_visibility="collapsed"
            )
            st.session_state.answer_mode = mode

            st.markdown("---")
            st.markdown(
                "<span class='sidebar-label'>Active Document</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class='file-pill'>
                    <span>📄</span>
                    <span class='file-pill-name'>{st.session_state.document_name}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "<span class='sidebar-label'>Session</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class='msg-count'>
                    <span class='msg-number'>{len(st.session_state.chat_history)}</span>
                    <span class='msg-label'>Messages</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown(
                "<span class='sidebar-label'>Quick Actions</span>",
                unsafe_allow_html=True
            )

            for action in [
                "📝 Summarize document",
                "✅ Extract action items",
                "⚠️ Identify risks",
                "📊 Extract key metrics",
                "❓ Generate FAQ"
            ]:
                if st.button(action, use_container_width=True, key=f"qa_{action[:10]}"):
                    with st.spinner(f"Working on: {action}..."):
                        fn = get_agent("action")
                        result = fn(
                            action=action,
                            context=st.session_state.document_text[:4000],
                            file_name=st.session_state.document_name
                        )
                        if result["success"]:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**{action}**\n\n{result['result']}"
                            })
                            st.rerun()

            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.memory.clear()
                    st.session_state.chat_history = []
                    st.rerun()
            with col_b:
                if st.button("📂 New", use_container_width=True):
                    for key in ["document_processed", "insights_loaded",
                                "entities_loaded", "map_loaded"]:
                        st.session_state[key] = False
                    for key in ["file_path", "document_name", "document_text"]:
                        st.session_state[key] = ""
                    st.session_state.entities = {}
                    st.session_state.insights = {}
                    st.session_state.document_map = []
                    st.session_state.chat_history = []
                    st.session_state.memory = deps["SessionMemory"]()
                    st.rerun()

    if not st.session_state.document_processed:
        show_hero()
        show_welcome_features()
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "🔍 Insights",
        "📊 Entities",
        "🗺️ Document Map",
        "✂️ Ask by Selection"
    ])

    with tab1:
        if len(st.session_state.multi_documents) > 1:
            with st.expander("🗂️ Multi-Document Query"):
                multi_q = st.text_input(
                    "Ask across all documents:",
                    placeholder="Compare topics across all uploaded files..."
                )
                if st.button("🔍 Query All Documents"):
                    if multi_q:
                        with st.spinner("Querying all documents..."):
                            fn = get_agent("multi_doc")
                            multi_answer = fn(
                                multi_q,
                                st.session_state.multi_documents
                            )
                            show_message("assistant", multi_answer)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": multi_answer
                            })

        for message in st.session_state.chat_history:
            show_message(message["role"], message["content"])

        col_input, col_voice = st.columns([6, 1])

        with col_input:
            question = st.chat_input("Ask anything about your document...")

        with col_voice:
            try:
                is_https = st.context.headers.get(
                    "x-forwarded-proto"
                ) == "https"
            except Exception:
                is_https = False

            if is_https:
                try:
                    from streamlit_mic_recorder import mic_recorder
                    audio = mic_recorder(
                        start_prompt="🎤",
                        stop_prompt="⏹️",
                        key="voice"
                    )
                    if audio and audio.get("bytes"):
                        with st.spinner("Transcribing..."):
                            fn = get_agent("voice")
                            voice_result = fn(audio["bytes"])
                            if voice_result["success"]:
                                question = voice_result["text"]
                                st.success(f"🎤 {question}")
                            else:
                                st.warning(voice_result["error"])
                except Exception:
                    st.button("🎤", help="Voice unavailable")
            else:
                if st.button("🎤", help="Voice works after HTTPS deployment"):
                    st.session_state.show_voice_tip = True

        if st.session_state.get("show_voice_tip"):
            st.info(
                "🎤 Voice works automatically after deploying to Streamlit Cloud with HTTPS."
            )
            if st.button("Got it ✓"):
                st.session_state.show_voice_tip = False
                st.rerun()

        if question:
            handle_question(question, deps)

    with tab2:
        st.markdown("### 🔍 Living Insight Layer")
        if not st.session_state.insights_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-content'>Click below to generate AI-powered insights from your document including summary, key findings, risks, and recommended next steps.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🧠 Generate Insights", use_container_width=True):
                with st.spinner("Generating insights..."):
                    fn = get_agent("insight")
                    st.session_state.insights = fn(
                        st.session_state.document_text[:4000]
                    )
                    st.session_state.insights_loaded = True
                    st.rerun()
        else:
            insights = st.session_state.insights
            icon_map = {
                "SUMMARY": ("📄", "Document Summary"),
                "KEY_FINDINGS": ("🎯", "Key Findings"),
                "ACTION_ITEMS": ("✅", "Action Items"),
                "RISKS": ("⚠️", "Risks and Concerns"),
                "DECISIONS": ("💡", "Key Decisions"),
                "NEXT_STEPS": ("🚀", "Recommended Next Steps")
            }
            for key, (icon, title) in icon_map.items():
                if insights.get(key):
                    st.markdown(
                        f"""
                        <div class='insight-card'>
                            <div class='insight-icon-title'>{icon} {title}</div>
                            <div class='insight-content'>{insights[key]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with tab3:
        st.markdown("### 📊 Entity Extraction Dashboard")
        if not st.session_state.entities_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-content'>Click below to extract people, organizations, dates, locations, and key terms from your document.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🔍 Extract Entities", use_container_width=True):
                with st.spinner("Extracting entities..."):
                    fn = get_agent("entity")
                    st.session_state.entities = fn(
                        st.session_state.document_text[:3000]
                    )
                    st.session_state.entities_loaded = True
                    st.rerun()
        else:
            entities = st.session_state.entities
            entity_icons = {
                "PEOPLE": "👤",
                "ORGANIZATIONS": "🏢",
                "DATES": "📅",
                "LOCATIONS": "📍",
                "KEY_TERMS": "🔑",
                "ACTION_ITEMS": "✅"
            }
            for entity_type, values in entities.items():
                if values:
                    icon = entity_icons.get(entity_type, "•")
                    st.markdown(f"**{icon} {entity_type}**")
                    tags = "".join([
                        f"<span class='entity-tag'>{v}</span>"
                        for v in values
                    ])
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown("")

    with tab4:
        st.markdown("### 🗺️ Document Map")
        if not st.session_state.map_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-content'>Click below to build a smart navigation map of your document sections and headings.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🗺️ Build Document Map", use_container_width=True):
                with st.spinner("Building map..."):
                    fn = get_agent("document_map")
                    st.session_state.document_map = fn(
                        st.session_state.document_text[:5000]
                    )
                    st.session_state.map_loaded = True
                    st.rerun()
        else:
            for section in st.session_state.document_map:
                with st.expander(f"📍 {section['title']}"):
                    st.markdown(section.get("description", ""))
                    if st.button(
                        "Ask about this section",
                        key=f"map_{section['title'][:20]}"
                    ):
                        handle_question(
                            f"Tell me about: {section['title']}",
                            deps
                        )

    with tab5:
        st.markdown("### ✂️ Ask by Selection")
        st.markdown(
            "<div class='insight-card'><div class='insight-content'>Paste any specific text from your document and choose an action. The AI will analyze just that selected portion with laser precision.</div></div>",
            unsafe_allow_html=True
        )

        selected_text = st.text_area(
            "Paste text here:",
            placeholder="Paste any paragraph, sentence, table, or section from your document...",
            height=140
        )

        selected_action = st.selectbox(
            "Choose action:",
            options=[
                "Explain this in detail",
                "Simplify this",
                "Find issues or problems",
                "Summarize this",
                "Extract key points",
                "Rewrite professionally",
                "Translate to simple English"
            ]
        )

        if st.button("🔍 Analyze Selection", use_container_width=True):
            if selected_text.strip():
                with st.spinner("Analyzing..."):
                    fn = get_agent("selection")
                    result = fn(selected_text, selected_action)

                st.markdown(
                    f"""
                    <div class='insight-card'>
                        <div class='insight-icon-title'>✨ Analysis Result</div>
                        <div class='insight-content'>{result}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                export_as_txt, _ = get_export()
                path = export_as_txt(result, "selection")
                with open(path, "rb") as f:
                    st.download_button(
                        "📥 Download Result",
                        data=f,
                        file_name="selection_analysis.txt"
                    )
            else:
                st.warning("Please paste some text first.")


if __name__ == "__main__":
    main()