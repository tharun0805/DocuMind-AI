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


def get_agent(agent_name: str):
    if agent_name == "suggestions":
        from agents.suggestion_agent import generate_suggestions
        return generate_suggestions
    elif agent_name == "actions":
        from agents.action_suggestions_agent import suggest_next_actions
        return suggest_next_actions
    elif agent_name == "knowledge":
        from agents.knowledge_agent import expand_knowledge
        return expand_knowledge
    elif agent_name == "entity":
        from agents.entity_agent import extract_entities
        return extract_entities
    elif agent_name == "insight":
        from agents.insight_agent import generate_insights
        return generate_insights
    elif agent_name == "document_map":
        from agents.document_map_agent import extract_document_map
        return extract_document_map
    elif agent_name == "selection":
        from agents.selection_agent import ask_about_selection
        return ask_about_selection
    elif agent_name == "action":
        from agents.document_action_agent import perform_document_action
        return perform_document_action
    elif agent_name == "multi_doc":
        from agents.multi_document_agent import query_multiple_documents
        return query_multiple_documents
    elif agent_name == "voice":
        from agents.voice_agent import transcribe_audio_file
        return transcribe_audio_file


def get_export_tools():
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e6edf3;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, rgba(88,166,255,0.05), rgba(63,185,80,0.05));
        border-radius: 16px;
        border: 1px solid rgba(88,166,255,0.1);
        margin-bottom: 24px;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff 0%, #3fb950 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .main-header p { color: #8b949e; font-size: 1rem; }
    .sidebar-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(88,166,255,0.3) !important;
    }
    .file-info-card {
        background: rgba(63,185,80,0.08);
        border: 1px solid rgba(63,185,80,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .file-info-card p {
        color: #3fb950;
        font-size: 0.83rem;
        font-weight: 500;
        margin: 0;
    }
    .stats-card {
        background: rgba(88,166,255,0.05);
        border: 1px solid rgba(88,166,255,0.1);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 6px 0;
    }
    .stats-card h3 { color: #58a6ff; font-size: 1.4rem; font-weight: 700; margin: 0; }
    .stats-card p { color: #8b949e; font-size: 0.72rem; margin: 0; text-transform: uppercase; }
    .insight-card {
        background: rgba(63,185,80,0.05);
        border: 1px solid rgba(63,185,80,0.15);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .insight-card h4 { color: #3fb950; font-size: 0.88rem; font-weight: 600; margin-bottom: 6px; }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 24px;
    }
    .feature-card {
        background: rgba(22,27,34,0.8);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 18px;
        transition: all 0.2s ease;
    }
    .feature-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .feature-card .icon { font-size: 1.6rem; margin-bottom: 8px; }
    .feature-card h4 { color: #e6edf3; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
    .feature-card p { color: #8b949e; font-size: 0.78rem; margin: 0; line-height: 1.4; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    hr { border-color: #21262d !important; }
    </style>
    """, unsafe_allow_html=True)


def show_header():
    st.markdown("""
    <div class='main-header'>
        <h1>🧠 DocuMind AI</h1>
        <p>Intelligent Document Intelligence Platform — Powered by Gemini and LangGraph</p>
    </div>
    """, unsafe_allow_html=True)


def show_welcome():
    st.markdown("""
    <div style='text-align:center;margin-top:60px;padding:40px;'>
        <h2 style='color:#e6edf3;font-size:1.8rem;font-weight:600;'>
            Upload a document to get started
        </h2>
        <p style='color:#8b949e;margin-bottom:30px;'>
            Ask questions, get insights, and analyse your documents using AI
        </p>
        <div class='feature-grid'>
            <div class='feature-card'>
                <div class='icon'>📄</div>
                <h4>Multi-Format</h4>
                <p>PDF, Word, Excel, PowerPoint, CSV, TXT</p>
            </div>
            <div class='feature-card'>
                <div class='icon'>🔍</div>
                <h4>Hybrid Search</h4>
                <p>FAISS semantic + BM25 keyword search</p>
            </div>
            <div class='feature-card'>
                <div class='icon'>🤖</div>
                <h4>Multi-Agent AI</h4>
                <p>Intent, Planner, Retriever, QA agents</p>
            </div>
            <div class='feature-card'>
                <div class='icon'>💬</div>
                <h4>Memory</h4>
                <p>Full conversation memory per document</p>
            </div>
            <div class='feature-card'>
                <div class='icon'>📊</div>
                <h4>Data Analysis</h4>
                <p>Compute answers from Excel and CSV</p>
            </div>
            <div class='feature-card'>
                <div class='icon'>🔒</div>
                <h4>100% Private</h4>
                <p>Documents never leave your machine</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_chat_message(role: str, content: str):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)


def show_evidence(evidence: list):
    if not evidence:
        return
    with st.expander("📎 View Evidence Sources"):
        for i, chunk in enumerate(evidence, 1):
            st.markdown(
                f"""<div style='background:rgba(88,166,255,0.05);
                border-left:3px solid #58a6ff;border-radius:0 8px 8px 0;
                padding:10px 14px;margin-bottom:10px;font-size:0.83rem;
                color:#8b949e;line-height:1.5;'>
                <strong style='color:#58a6ff;'>Source {i}:</strong>
                <br>{chunk}</div>""",
                unsafe_allow_html=True
            )


def show_file_info(name: str, size: int):
    kb = size / 1024
    st.markdown(
        f"<div class='file-info-card'><p>📄 <strong>{name}</strong> — {kb:.1f} KB</p></div>",
        unsafe_allow_html=True
    )


def show_stats(count: int):
    st.markdown(
        f"<div class='stats-card'><h3>{count}</h3><p>Messages</p></div>",
        unsafe_allow_html=True
    )


def show_insight_card(title: str, content: str, icon: str):
    st.markdown(
        f"<div class='insight-card'><h4>{icon} {title}</h4>"
        f"<p style='color:#8b949e;font-size:0.83rem;line-height:1.5;'>{content}</p></div>",
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

        tracker = deps["PerformanceTracker"]()

        tracker.start("reading")
        with st.spinner("📖 Reading document..."):
            text = deps["load_document"](tmp_path)
            st.session_state.document_text = text
        tracker.end("reading")

        tracker.start("chunking")
        with st.spinner("✂️ Chunking..."):
            chunks = deps["chunk_text"](text)
        tracker.end("chunking")

        tracker.start("indexing")
        with st.spinner("🔢 Building search index..."):
            deps["create_vector_store"](chunks)
            deps["create_bm25_index"](chunks)
        tracker.end("indexing")

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

        doc_entry = {
            "name": uploaded_file.name,
            "path": tmp_path,
            "text": text
        }
        existing = [d["name"] for d in st.session_state.multi_documents]
        if uploaded_file.name not in existing:
            st.session_state.multi_documents.append(doc_entry)

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

    show_chat_message("human", question)
    st.session_state.chat_history.append({
        "role": "human",
        "content": question
    })

    with st.spinner("🧠 DocuMind is analysing your question..."):
        try:
            result = deps["run_workflow"](
                question=question,
                memory=st.session_state.memory,
                file_path=st.session_state.file_path,
                answer_mode=answer_mode
            )

            answer = result["answer"]
            evidence = result["evidence"]

            show_chat_message("assistant", answer)
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

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "💡 Suggestions",
            key=f"sug_{len(st.session_state.chat_history)}"
        ):
            with st.spinner("Generating..."):
                fn = get_agent("suggestions")
                suggestions = fn(question, answer)
                for s in suggestions:
                    st.markdown(f"- {s}")

    with col2:
        if st.button(
            "🎓 Learn More",
            key=f"know_{len(st.session_state.chat_history)}"
        ):
            with st.spinner("Expanding..."):
                fn = get_agent("knowledge")
                knowledge = fn(question, answer)
                if knowledge.get("YOUTUBE_SEARCHES"):
                    st.markdown("**📺 YouTube:**")
                    for s in knowledge["YOUTUBE_SEARCHES"]:
                        st.markdown(f"- {s}")
                if knowledge.get("RELATED_TOPICS"):
                    st.markdown("**🔗 Topics:**")
                    for t in knowledge["RELATED_TOPICS"]:
                        st.markdown(f"- {t}")

    with col3:
        export_as_txt, _ = get_export_tools()
        txt_path = export_as_txt(
            answer,
            f"answer_{len(st.session_state.chat_history)}"
        )
        with open(txt_path, "rb") as f:
            st.download_button(
                "📥 Download",
                data=f,
                file_name="documind_answer.txt",
                key=f"dl_{len(st.session_state.chat_history)}"
            )


def main():
    apply_css()
    show_header()

    with st.spinner("⚡ Starting DocuMind AI..."):
        deps = load_core_dependencies()

    initialize_session(deps)

    with st.sidebar:
        st.markdown(
            "<p class='sidebar-title'>Document Upload</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            show_file_info(uploaded_file.name, uploaded_file.size)
            if st.button("🚀 Process Document", use_container_width=True):
                success = process_document(uploaded_file, deps)
                if success:
                    st.success("✅ Document ready!")
                    st.rerun()

        if st.session_state.document_processed:
            st.markdown("---")
            st.markdown(
                "<p class='sidebar-title'>Answer Mode</p>",
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
                "<p class='sidebar-title'>Current Document</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='file-info-card'><p>📄 {st.session_state.document_name}</p></div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p class='sidebar-title'>Session Stats</p>",
                unsafe_allow_html=True
            )
            show_stats(len(st.session_state.chat_history))

            st.markdown("---")
            st.markdown(
                "<p class='sidebar-title'>Quick Actions</p>",
                unsafe_allow_html=True
            )

            quick_actions = [
                "📝 Summarize entire document",
                "✅ Extract all action items",
                "⚠️ Identify all risks",
                "📊 Extract key metrics",
                "❓ Generate FAQ",
                "🔄 Rewrite for general audience"
            ]

            for action in quick_actions:
                if st.button(
                    action,
                    use_container_width=True,
                    key=f"qa_{action[:12]}"
                ):
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

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.chat_history = []
                st.rerun()

            if st.button("📂 New Document", use_container_width=True):
                for key in [
                    "document_processed", "file_path", "document_name",
                    "document_text", "insights_loaded", "entities_loaded",
                    "map_loaded"
                ]:
                    st.session_state[key] = (
                        False if isinstance(st.session_state[key], bool)
                        else ""
                    )
                st.session_state.entities = {}
                st.session_state.insights = {}
                st.session_state.document_map = []
                st.session_state.chat_history = []
                st.session_state.memory = deps["SessionMemory"]()
                st.rerun()

    if not st.session_state.document_processed:
        show_welcome()
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
                    placeholder="Compare the main topics..."
                )
                if st.button("🔍 Query All"):
                    if multi_q:
                        with st.spinner("Querying..."):
                            fn = get_agent("multi_doc")
                            multi_answer = fn(
                                multi_q,
                                st.session_state.multi_documents
                            )
                            show_chat_message("assistant", multi_answer)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": multi_answer
                            })

        for message in st.session_state.chat_history:
            show_chat_message(message["role"], message["content"])

        col_input, col_voice = st.columns([6, 1])

        with col_input:
            question = st.chat_input("Ask anything about your document...")

        with col_voice:
            try:
                is_https = (
                    st.context.headers.get("x-forwarded-proto") == "https"
                )
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
                "🎤 Voice input requires HTTPS. "
                "It will work automatically after deployment to Streamlit Cloud. "
                "For now please type your question."
            )
            if st.button("Got it ✓"):
                st.session_state.show_voice_tip = False
                st.rerun()

        if question:
            handle_question(question, deps)

    with tab2:
        st.markdown("### 🔍 Living Insight Layer")
        if not st.session_state.insights_loaded:
            st.info("Click below to generate insights from your document.")
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
                "RISKS": ("⚠️", "Risks"),
                "DECISIONS": ("💡", "Key Decisions"),
                "NEXT_STEPS": ("🚀", "Next Steps")
            }
            for key, (icon, title) in icon_map.items():
                if insights.get(key):
                    show_insight_card(title, insights[key], icon)

    with tab3:
        st.markdown("### 📊 Entity Extraction Dashboard")
        if not st.session_state.entities_loaded:
            st.info("Click below to extract entities from your document.")
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
            for entity_type, values in entities.items():
                if values:
                    st.markdown(f"**{entity_type}:**")
                    tags = "".join([
                        f"<span style='display:inline-block;"
                        f"background:rgba(210,153,34,0.1);"
                        f"border:1px solid rgba(210,153,34,0.3);"
                        f"border-radius:6px;padding:3px 10px;"
                        f"margin:3px;font-size:0.8rem;"
                        f"color:#d29922;'>{v}</span>"
                        for v in values
                    ])
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown("")

    with tab4:
        st.markdown("### 🗺️ Document Map")
        if not st.session_state.map_loaded:
            st.info("Click below to build a navigation map of your document.")
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
                        "Ask about this",
                        key=f"map_{section['title'][:20]}"
                    ):
                        handle_question(
                            f"Tell me about: {section['title']}",
                            deps
                        )

    with tab5:
        st.markdown("### ✂️ Ask by Selection")
        st.markdown("Paste any text from the document and ask about it.")

        selected_text = st.text_area(
            "Paste text here:",
            placeholder="Paste any paragraph or section...",
            height=130
        )

        selected_action = st.selectbox(
            "Action:",
            options=[
                "Explain this",
                "Simplify this",
                "Find issues",
                "Summarize this",
                "Key points here?",
                "Rewrite professionally"
            ]
        )

        if st.button("🔍 Analyze", use_container_width=True):
            if selected_text.strip():
                with st.spinner("Analyzing..."):
                    fn = get_agent("selection")
                    result = fn(selected_text, selected_action)
                    st.markdown("### Result:")
                    st.markdown(result)
                    export_as_txt, _ = get_export_tools()
                    path = export_as_txt(result, "selection")
                    with open(path, "rb") as f:
                        st.download_button(
                            "📥 Download",
                            data=f,
                            file_name="selection.txt"
                        )
            else:
                st.warning("Please paste some text first.")


if __name__ == "__main__":
    main()