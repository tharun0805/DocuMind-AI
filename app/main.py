import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
import random
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
    import importlib
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
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    return None


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
        "multi_documents": [],
        "insights_loaded": False,
        "entities_loaded": False,
        "map_loaded": False,
        "show_voice_tip": False,
        "last_question": "",
        "last_answer": "",
        "suggested_prompts": [],
        "quiz_questions": [],
        "quiz_loaded": False,
        "tts_enabled": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }

    .stApp {
        background: #020409;
        min-height: 100vh;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% -20%, rgba(88,166,255,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(63,185,80,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 60% at 50% 50%, rgba(139,92,246,0.05) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    section[data-testid="stSidebar"] {
        background: rgba(2,4,9,0.97) !important;
        border-right: 1px solid rgba(88,166,255,0.06) !important;
        backdrop-filter: blur(40px);
    }

    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    .hero {
        text-align: center;
        padding: 80px 40px 60px;
        position: relative;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 100px;
        padding: 8px 20px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -2px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #58a6ff 40%, #3fb950 70%, #ffffff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .hero-sub {
        font-size: 1.15rem;
        color: #8b949e;
        max-width: 560px;
        margin: 0 auto 48px;
        line-height: 1.7;
        font-weight: 400;
    }

    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 0;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        overflow: hidden;
        max-width: 600px;
        margin: 0 auto;
        background: rgba(22,27,34,0.4);
        backdrop-filter: blur(20px);
    }

    .stat {
        flex: 1;
        padding: 20px;
        border-right: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }

    .stat:last-child { border-right: none; }

    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
    }

    .stat-lbl {
        font-size: 0.68rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        display: block;
    }

    .feat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        padding: 0 40px 60px;
    }

    .feat-card {
        background: rgba(13,17,23,0.6);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .feat-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(88,166,255,0.03), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .feat-card:hover {
        border-color: rgba(88,166,255,0.15);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(88,166,255,0.08);
    }

    .feat-card:hover::after { opacity: 1; }

    .feat-emoji { font-size: 1.8rem; margin-bottom: 12px; display: block; }
    .feat-name { color: #e6edf3; font-size: 0.88rem; font-weight: 600; margin-bottom: 6px; }
    .feat-text { color: #8b949e; font-size: 0.76rem; line-height: 1.5; }

    .sidebar-lbl {
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        color: rgba(139,148,158,0.6) !important;
        text-transform: uppercase !important;
        letter-spacing: 2.5px !important;
        margin-bottom: 10px !important;
        display: block !important;
    }

    .doc-pill {
        background: rgba(63,185,80,0.07);
        border: 1px solid rgba(63,185,80,0.15);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
    }

    .doc-name { color: #3fb950; font-size: 0.82rem; font-weight: 600; }
    .doc-size { color: #8b949e; font-size: 0.72rem; }

    .counter {
        background: rgba(88,166,255,0.05);
        border: 1px solid rgba(88,166,255,0.1);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }

    .counter-num {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
    }

    .counter-lbl {
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 12px rgba(31,111,235,0.25) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(88,166,255,0.3) !important;
    }

    .prompt-chip {
        display: inline-block;
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 100px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #58a6ff;
        cursor: pointer;
        margin: 3px;
        transition: all 0.2s ease;
        white-space: nowrap;
    }

    .prompt-chip:hover {
        background: rgba(88,166,255,0.12);
        border-color: rgba(88,166,255,0.3);
        transform: translateY(-1px);
    }

    .insight-card {
        background: rgba(13,17,23,0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }

    .insight-card:hover {
        border-color: rgba(63,185,80,0.2);
        background: rgba(63,185,80,0.03);
    }

    .insight-head {
        color: #3fb950;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .insight-body {
        color: #8b949e;
        font-size: 0.82rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .entity-tag {
        display: inline-block;
        background: rgba(210,153,34,0.07);
        border: 1px solid rgba(210,153,34,0.18);
        border-radius: 8px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.77rem;
        color: #e3b341;
        transition: all 0.15s ease;
    }

    .entity-tag:hover {
        background: rgba(210,153,34,0.13);
        transform: scale(1.02);
    }

    .evidence-item {
        background: rgba(88,166,255,0.04);
        border-left: 3px solid rgba(88,166,255,0.4);
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.81rem;
        color: #8b949e;
        line-height: 1.55;
    }

    .resource-link {
        background: rgba(22,27,34,0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }

    .resource-link:hover {
        border-color: rgba(88,166,255,0.15);
        transform: translateX(3px);
    }

    .yt-link {
        background: rgba(255,60,60,0.05);
        border: 1px solid rgba(255,60,60,0.12);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }

    .yt-link:hover {
        border-color: rgba(255,60,60,0.25);
        background: rgba(255,60,60,0.08);
    }

    .quiz-card {
        background: rgba(139,92,246,0.05);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .quiz-q {
        color: #e6edf3;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .quiz-opt {
        background: rgba(139,92,246,0.06);
        border: 1px solid rgba(139,92,246,0.12);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #c9d1d9;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .quiz-opt:hover {
        background: rgba(139,92,246,0.12);
        border-color: rgba(139,92,246,0.25);
    }

    .tts-bar {
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.12);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .stChatInputContainer {
        background: rgba(13,17,23,0.8) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(20px) !important;
    }

    .stChatInputContainer:focus-within {
        border-color: rgba(88,166,255,0.3) !important;
        box-shadow: 0 0 0 4px rgba(88,166,255,0.06) !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(13,17,23,0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(88,166,255,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(88,166,255,0.3); }

    hr { border-color: rgba(255,255,255,0.05) !important; }
    </style>
    """, unsafe_allow_html=True)


def generate_suggested_prompts(text: str) -> list[str]:
    try:
        from utils.llm_provider import get_shared_llm
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Based on this document generate exactly 6 specific and interesting
            questions a user would want to ask about it.

            Document: {text}

            Rules:
            - Make questions specific to the document content
            - Mix factual, analytical, and summary questions
            - Keep each question under 10 words
            - One question per line
            - No numbering or bullets

            Questions:
            """
        )
        llm = get_shared_llm(temperature=0.5)
        chain = prompt | llm
        result = chain.invoke({"text": text[:2000]})
        lines = [
            l.strip() for l in result.content.strip().split("\n")
            if l.strip() and len(l.strip()) > 5
        ]
        return lines[:6]
    except Exception:
        return [
            "What is this document about?",
            "Summarize the key points",
            "What are the main findings?",
            "What actions are recommended?",
            "What are the risks mentioned?",
            "Who is this document for?"
        ]


def generate_quiz(text: str) -> list[dict]:
    try:
        from utils.llm_provider import get_shared_llm
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Generate exactly 5 multiple choice quiz questions about this document.

            Document: {text}

            For each question use this exact format:
            Q: [question]
            A: [correct answer]
            B: [wrong option]
            C: [wrong option]
            D: [wrong option]
            ANSWER: A

            Generate 5 questions separated by blank lines.
            """
        )
        llm = get_shared_llm(temperature=0.3)
        chain = prompt | llm
        result = chain.invoke({"text": text[:3000]})

        questions = []
        blocks = result.content.strip().split("\n\n")

        for block in blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            q_data = {}
            options = []
            for line in lines:
                if line.startswith("Q:"):
                    q_data["question"] = line[2:].strip()
                elif line.startswith("A:"):
                    options.append(("A", line[2:].strip()))
                elif line.startswith("B:"):
                    options.append(("B", line[2:].strip()))
                elif line.startswith("C:"):
                    options.append(("C", line[2:].strip()))
                elif line.startswith("D:"):
                    options.append(("D", line[2:].strip()))
                elif line.startswith("ANSWER:"):
                    q_data["answer"] = line[7:].strip()

            if q_data.get("question") and options:
                q_data["options"] = options
                questions.append(q_data)

        return questions[:5]
    except Exception:
        return []


def text_to_speech_js(text: str) -> str:
    clean = text.replace('"', '\\"').replace('\n', ' ').replace("'", "\\'")
    return f"""
    <script>
    function speakText() {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{clean[:500]}");
        msg.rate = 0.9;
        msg.pitch = 1;
        msg.volume = 1;
        window.speechSynthesis.speak(msg);
    }}
    function stopSpeech() {{
        window.speechSynthesis.cancel();
    }}
    </script>
    <div style='display:flex;gap:10px;margin-top:10px;'>
        <button onclick='speakText()' style='
            background:rgba(88,166,255,0.1);
            border:1px solid rgba(88,166,255,0.2);
            border-radius:8px;
            color:#58a6ff;
            padding:6px 14px;
            font-size:0.8rem;
            cursor:pointer;
            font-family:Inter,sans-serif;
        '>🔊 Read Aloud</button>
        <button onclick='stopSpeech()' style='
            background:rgba(248,81,73,0.08);
            border:1px solid rgba(248,81,73,0.15);
            border-radius:8px;
            color:#f85149;
            padding:6px 14px;
            font-size:0.8rem;
            cursor:pointer;
            font-family:Inter,sans-serif;
        '>⏹ Stop</button>
    </div>
    """


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

        steps = [
            (20, "📖 Reading document..."),
            (45, "✂️ Chunking intelligently..."),
            (70, "🔢 Building vector index..."),
            (90, "🔍 Building keyword index..."),
            (100, "✅ Complete!")
        ]

        progress = st.progress(0, text="Starting...")

        progress.progress(10, text="📖 Reading document...")
        text = deps["load_document"](tmp_path)
        st.session_state.document_text = text

        progress.progress(35, text="✂️ Chunking intelligently...")
        chunks = deps["chunk_text"](text)

        progress.progress(60, text="🔢 Building vector index...")
        deps["create_vector_store"](chunks)

        progress.progress(85, text="🔍 Building keyword index...")
        deps["create_bm25_index"](chunks)

        progress.progress(95, text="💡 Generating smart prompts...")
        st.session_state.suggested_prompts = generate_suggested_prompts(text)

        progress.progress(100, text="✅ Ready!")
        time.sleep(0.3)
        progress.empty()

        st.session_state.document_processed = True
        st.session_state.insights_loaded = False
        st.session_state.entities_loaded = False
        st.session_state.map_loaded = False
        st.session_state.quiz_loaded = False
        st.session_state.entities = {}
        st.session_state.insights = {}
        st.session_state.document_map = []
        st.session_state.quiz_questions = []
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


def show_message(role: str, content: str):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)
            st.components.v1.html(text_to_speech_js(content), height=60)


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

    with st.spinner("🧠 Thinking..."):
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

            if evidence:
                with st.expander("📎 Evidence Sources"):
                    for i, chunk in enumerate(evidence, 1):
                        st.markdown(
                            f"<div class='evidence-item'><strong style='color:#58a6ff;'>Source {i}</strong><br>{chunk}</div>",
                            unsafe_allow_html=True
                        )

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
        if st.button("💡 Suggestions", key=f"sug_{len(st.session_state.chat_history)}", use_container_width=True):
            fn = get_agent("suggestions")
            with st.spinner("..."):
                s = fn(question, answer)
            for item in s:
                st.markdown(f"• {item}")

    with col2:
        if st.button("🌐 Resources", key=f"res_{len(st.session_state.chat_history)}", use_container_width=True):
            fn = get_agent("knowledge")
            with st.spinner("Finding..."):
                k = fn(question, answer)
            if k:
                if k.get("YOUTUBE_SEARCHES"):
                    st.markdown("**📺 YouTube:**")
                    for s in k["YOUTUBE_SEARCHES"]:
                        url = f"https://www.youtube.com/results?search_query={s.replace(' ', '+')}"
                        st.markdown(
                            f"<div class='yt-link'><a href='{url}' target='_blank'>▶ {s}</a></div>",
                            unsafe_allow_html=True
                        )
                if k.get("RELATED_TOPICS"):
                    st.markdown("**🔗 Topics:**")
                    for t in k["RELATED_TOPICS"]:
                        url = f"https://www.google.com/search?q={t.replace(' ', '+')}"
                        st.markdown(
                            f"<div class='resource-link'><a href='{url}' target='_blank' style='color:#58a6ff;text-decoration:none;'>🔍 {t}</a></div>",
                            unsafe_allow_html=True
                        )
                if k.get("SIMILAR_RESOURCES"):
                    st.markdown("**📚 Resources:**")
                    for r in k["SIMILAR_RESOURCES"]:
                        url = f"https://scholar.google.com/scholar?q={r.replace(' ', '+')}"
                        st.markdown(
                            f"<div class='resource-link'><a href='{url}' target='_blank' style='color:#58a6ff;text-decoration:none;'>🎓 {r}</a></div>",
                            unsafe_allow_html=True
                        )

    with col3:
        from tools.file_export_tool import export_as_txt
        path = export_as_txt(answer, f"ans_{len(st.session_state.chat_history)}")
        with open(path, "rb") as f:
            st.download_button(
                "📥 Download",
                data=f,
                file_name="answer.txt",
                use_container_width=True,
                key=f"dl_{len(st.session_state.chat_history)}"
            )


def main():
    apply_css()

    with st.spinner("⚡ Initializing..."):
        deps = load_core_dependencies()

    initialize_session(deps)

    with st.sidebar:
        st.markdown(
            "<span class='sidebar-lbl'>Upload Document</span>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "file",
            type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            kb = uploaded_file.size / 1024
            st.markdown(
                f"<div class='doc-pill'><div class='doc-name'>📄 {uploaded_file.name}</div><div class='doc-size'>{kb:.0f} KB</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🚀 Process", use_container_width=True):
                success = process_document(uploaded_file, deps)
                if success:
                    st.success("✅ Ready!")
                    st.rerun()

        if st.session_state.document_processed:
            st.markdown("---")
            st.markdown(
                "<span class='sidebar-lbl'>Answer Style</span>",
                unsafe_allow_html=True
            )
            mode = st.selectbox(
                "m",
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
                "<span class='sidebar-lbl'>Active File</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='doc-pill'><div class='doc-name'>📄 {st.session_state.document_name}</div></div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<span class='sidebar-lbl'>Session</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='counter'><span class='counter-num'>{len(st.session_state.chat_history)}</span><span class='counter-lbl'>Messages</span></div>",
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown(
                "<span class='sidebar-lbl'>Quick Actions</span>",
                unsafe_allow_html=True
            )

            for action in [
                "📝 Summarize document",
                "✅ Extract action items",
                "⚠️ Identify all risks",
                "📊 Extract key metrics",
                "❓ Generate FAQ",
                "🔄 Rewrite simply"
            ]:
                if st.button(action, use_container_width=True, key=f"qa_{action[:8]}"):
                    with st.spinner("Working..."):
                        fn = get_agent("action")
                        r = fn(
                            action=action,
                            context=st.session_state.document_text[:4000],
                            file_name=st.session_state.document_name
                        )
                        if r["success"]:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**{action}**\n\n{r['result']}"
                            })
                            st.rerun()

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.memory.clear()
                    st.session_state.chat_history = []
                    st.rerun()
            with c2:
                if st.button("📂 New", use_container_width=True):
                    for k in ["document_processed", "insights_loaded", "entities_loaded", "map_loaded", "quiz_loaded"]:
                        st.session_state[k] = False
                    for k in ["file_path", "document_name", "document_text"]:
                        st.session_state[k] = ""
                    st.session_state.entities = {}
                    st.session_state.insights = {}
                    st.session_state.document_map = []
                    st.session_state.chat_history = []
                    st.session_state.suggested_prompts = []
                    st.session_state.memory = deps["SessionMemory"]()
                    st.rerun()

    if not st.session_state.document_processed:
        st.markdown("""
        <div class='hero'>
            <div class='hero-eyebrow'>✦ Next-Gen Document Intelligence</div>
            <div class='hero-title'>DocuMind AI</div>
            <div class='hero-sub'>Upload any document. Ask anything. Get instant expert-level answers powered by multi-agent AI, hybrid search, and real-time knowledge expansion.</div>
            <div class='hero-stats'>
                <div class='stat'><span class='stat-val'>6+</span><span class='stat-lbl'>Formats</span></div>
                <div class='stat'><span class='stat-val'>5</span><span class='stat-lbl'>AI Agents</span></div>
                <div class='stat'><span class='stat-val'>20+</span><span class='stat-lbl'>Features</span></div>
                <div class='stat'><span class='stat-val'>100%</span><span class='stat-lbl'>Private</span></div>
            </div>
        </div>
        <div class='feat-grid'>
            <div class='feat-card'><span class='feat-emoji'>⚡</span><div class='feat-name'>Ultra-Fast AI</div><div class='feat-text'>Groq-powered responses in under 2 seconds</div></div>
            <div class='feat-card'><span class='feat-emoji'>🔍</span><div class='feat-name'>Hybrid Search</div><div class='feat-text'>FAISS semantic + BM25 keyword with RRF reranking</div></div>
            <div class='feat-card'><span class='feat-emoji'>🧠</span><div class='feat-name'>Multi-Agent</div><div class='feat-text'>5 specialized agents orchestrated by LangGraph</div></div>
            <div class='feat-card'><span class='feat-emoji'>💬</span><div class='feat-name'>Memory</div><div class='feat-text'>Per-document conversation memory</div></div>
            <div class='feat-card'><span class='feat-emoji'>🔊</span><div class='feat-name'>Read Aloud</div><div class='feat-text'>Text-to-speech for every answer</div></div>
            <div class='feat-card'><span class='feat-emoji'>🎯</span><div class='feat-name'>Smart Prompts</div><div class='feat-text'>Auto-generated questions from your document</div></div>
            <div class='feat-card'><span class='feat-emoji'>🧩</span><div class='feat-name'>Quiz Mode</div><div class='feat-text'>Auto-generated quizzes from document content</div></div>
            <div class='feat-card'><span class='feat-emoji'>🌐</span><div class='feat-name'>Knowledge Links</div><div class='feat-text'>Real YouTube and web resource links</div></div>
        </div>
        """, unsafe_allow_html=True)
        return

    tabs = st.tabs([
        "💬 Chat",
        "💡 Smart Prompts",
        "🧩 Quiz",
        "🔍 Insights",
        "📊 Entities",
        "🗺️ Map",
        "✂️ Selection"
    ])

    with tabs[0]:
        if len(st.session_state.multi_documents) > 1:
            with st.expander("🗂️ Multi-Document Query"):
                mq = st.text_input("Ask across all documents:", placeholder="Compare topics...")
                if st.button("🔍 Query All"):
                    if mq:
                        with st.spinner("Querying..."):
                            fn = get_agent("multi_doc")
                            ma = fn(mq, st.session_state.multi_documents)
                            show_message("assistant", ma)
                            st.session_state.chat_history.append({"role": "assistant", "content": ma})

        for msg in st.session_state.chat_history:
            show_message(msg["role"], msg["content"])

        col_in, col_v = st.columns([6, 1])
        with col_in:
            question = st.chat_input("Ask anything about your document...")
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
                        with st.spinner("Transcribing..."):
                            fn = get_agent("voice")
                            vr = fn(audio["bytes"])
                            if vr["success"]:
                                question = vr["text"]
                                st.success(f"🎤 {question}")
                except Exception:
                    st.button("🎤", help="Voice unavailable")
            else:
                if st.button("🎤", help="Voice works after HTTPS deployment"):
                    st.info("Voice works on HTTPS deployment (Streamlit Cloud)")

        if question:
            handle_question(question, deps)

    with tabs[1]:
        st.markdown("### 💡 Smart Prompts")
        st.markdown("These questions were automatically generated from your document.")

        prompts = st.session_state.suggested_prompts
        if prompts:
            cols = st.columns(2)
            for i, prompt in enumerate(prompts):
                with cols[i % 2]:
                    if st.button(
                        f"💬 {prompt}",
                        key=f"prompt_{i}",
                        use_container_width=True
                    ):
                        st.session_state.active_tab = "chat"
                        handle_question(prompt, deps)
        else:
            if st.button("💡 Generate Smart Prompts", use_container_width=True):
                with st.spinner("Analyzing document..."):
                    st.session_state.suggested_prompts = generate_suggested_prompts(
                        st.session_state.document_text
                    )
                    st.rerun()

    with tabs[2]:
        st.markdown("### 🧩 Document Quiz")
        st.markdown("Test your understanding of the document.")

        if not st.session_state.quiz_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-body'>Click below to generate a 5-question multiple choice quiz based on your document content.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🧩 Generate Quiz", use_container_width=True):
                with st.spinner("Creating quiz questions..."):
                    st.session_state.quiz_questions = generate_quiz(
                        st.session_state.document_text
                    )
                    st.session_state.quiz_loaded = True
                    st.rerun()
        else:
            quiz = st.session_state.quiz_questions
            if quiz:
                for i, q in enumerate(quiz, 1):
                    st.markdown(
                        f"""
                        <div class='quiz-card'>
                            <div class='quiz-q'>Q{i}. {q.get('question', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if q.get("options"):
                        user_ans = st.radio(
                            f"Select answer for Q{i}:",
                            options=[f"{opt[0]}. {opt[1]}" for opt in q["options"]],
                            key=f"quiz_{i}",
                            label_visibility="collapsed"
                        )
                        if st.button(f"Check Q{i}", key=f"check_{i}"):
                            correct = q.get("answer", "A")
                            if user_ans and user_ans.startswith(correct):
                                st.success("✅ Correct!")
                            else:
                                correct_opt = next(
                                    (f"{o[0]}. {o[1]}" for o in q["options"] if o[0] == correct),
                                    correct
                                )
                                st.error(f"❌ Correct answer: {correct_opt}")
            else:
                st.warning("Could not generate quiz. Please try again.")
                if st.button("Retry Quiz"):
                    st.session_state.quiz_loaded = False
                    st.rerun()

    with tabs[3]:
        st.markdown("### 🔍 Living Insight Layer")
        if not st.session_state.insights_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-body'>Generate AI-powered insights including summary, key findings, risks, action items, and recommended next steps.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🧠 Generate Insights", use_container_width=True):
                with st.spinner("Analyzing..."):
                    fn = get_agent("insight")
                    st.session_state.insights = fn(st.session_state.document_text[:4000])
                    st.session_state.insights_loaded = True
                    st.rerun()
        else:
            icon_map = {
                "SUMMARY": ("📄", "Document Summary"),
                "KEY_FINDINGS": ("🎯", "Key Findings"),
                "ACTION_ITEMS": ("✅", "Action Items"),
                "RISKS": ("⚠️", "Risks"),
                "DECISIONS": ("💡", "Key Decisions"),
                "NEXT_STEPS": ("🚀", "Next Steps")
            }
            for key, (icon, title) in icon_map.items():
                if st.session_state.insights.get(key):
                    st.markdown(
                        f"<div class='insight-card'><div class='insight-head'>{icon} {title}</div><div class='insight-body'>{st.session_state.insights[key]}</div></div>",
                        unsafe_allow_html=True
                    )

    with tabs[4]:
        st.markdown("### 📊 Entity Dashboard")
        if not st.session_state.entities_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-body'>Extract people, organizations, dates, locations, and key terms from your document.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🔍 Extract Entities", use_container_width=True):
                with st.spinner("Extracting..."):
                    fn = get_agent("entity")
                    st.session_state.entities = fn(st.session_state.document_text[:3000])
                    st.session_state.entities_loaded = True
                    st.rerun()
        else:
            icons = {"PEOPLE": "👤", "ORGANIZATIONS": "🏢", "DATES": "📅", "LOCATIONS": "📍", "KEY_TERMS": "🔑", "ACTION_ITEMS": "✅"}
            for etype, vals in st.session_state.entities.items():
                if vals:
                    icon = icons.get(etype, "•")
                    st.markdown(f"**{icon} {etype}**")
                    tags = "".join([f"<span class='entity-tag'>{v}</span>" for v in vals])
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown("")

    with tabs[5]:
        st.markdown("### 🗺️ Document Map")
        if not st.session_state.map_loaded:
            st.markdown(
                "<div class='insight-card'><div class='insight-body'>Build a smart navigation map of your document structure.</div></div>",
                unsafe_allow_html=True
            )
            if st.button("🗺️ Build Map", use_container_width=True):
                with st.spinner("Mapping..."):
                    fn = get_agent("document_map")
                    st.session_state.document_map = fn(st.session_state.document_text[:5000])
                    st.session_state.map_loaded = True
                    st.rerun()
        else:
            for section in st.session_state.document_map:
                with st.expander(f"📍 {section['title']}"):
                    st.markdown(section.get("description", ""))
                    if st.button("Ask about this", key=f"map_{section['title'][:15]}"):
                        handle_question(f"Tell me about: {section['title']}", deps)

    with tabs[6]:
        st.markdown("### ✂️ Ask by Selection")
        st.markdown(
            "<div class='insight-card'><div class='insight-body'>Paste any specific text from the document and analyze it with laser precision.</div></div>",
            unsafe_allow_html=True
        )
        sel_text = st.text_area("Paste text here:", placeholder="Any paragraph or section...", height=130)
        sel_action = st.selectbox("Action:", ["Explain this", "Simplify this", "Find issues", "Summarize", "Key points", "Rewrite professionally"])

        if st.button("🔍 Analyze", use_container_width=True):
            if sel_text.strip():
                with st.spinner("Analyzing..."):
                    fn = get_agent("selection")
                    result = fn(sel_text, sel_action)
                st.markdown(
                    f"<div class='insight-card'><div class='insight-head'>✨ Result</div><div class='insight-body'>{result}</div></div>",
                    unsafe_allow_html=True
                )
                st.components.v1.html(text_to_speech_js(result), height=60)
            else:
                st.warning("Paste some text first.")


if __name__ == "__main__":
    main()