import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from ingestion.document_loader import load_document
from chunking.text_chunker import chunk_text
from vector_store.faiss_store import create_vector_store
from vector_store.bm25_store import create_bm25_index
from graph.workflow import run_workflow
from memory.session_memory import SessionMemory
from memory.file_memory_manager import FileMemoryManager
from agents.clarification_agent import needs_clarification
from agents.suggestion_agent import generate_suggestions
from agents.action_suggestions_agent import suggest_next_actions
from agents.entity_agent import extract_entities
from agents.insight_agent import generate_insights
from agents.document_action_agent import perform_document_action
from agents.knowledge_agent import expand_knowledge
from agents.voice_agent import transcribe_audio_file
from agents.document_map_agent import extract_document_map
from agents.selection_agent import ask_about_selection
from agents.multi_document_agent import query_multiple_documents
from tools.file_export_tool import export_as_txt, export_as_docx
from utils.validator import validate_file, validate_question
from utils.error_handler import handle_error
from utils.performance import PerformanceTracker
from app.ui_components import (
    apply_custom_css,
    show_header,
    show_welcome_screen,
    show_chat_message,
    show_success,
    show_error,
    show_warning,
    show_file_info,
    show_stats,
    show_thinking,
    show_answer_mode_selector,
    show_evidence_panel,
    show_insight_card
)


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session():
    defaults = {
        "memory": SessionMemory(),
        "file_memory_manager": FileMemoryManager(),
        "chat_history": [],
        "file_path": "",
        "document_processed": False,
        "document_name": "",
        "document_text": "",
        "entities": {},
        "insights": {},
        "document_map": [],
        "answer_mode": "detailed",
        "pending_question": None,
        "clarification_questions": [],
        "awaiting_clarification": False,
        "multi_documents": [],
        "insights_loaded": False,
        "entities_loaded": False,
        "map_loaded": False,
        "show_voice_tip": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def cached_extract_entities(text_sample: str) -> dict:
    return extract_entities(text_sample)


@st.cache_data(show_spinner=False)
def cached_generate_insights(text_sample: str) -> dict:
    return generate_insights(text_sample)


@st.cache_data(show_spinner=False)
def cached_document_map(text_sample: str) -> list:
    return extract_document_map(text_sample)


@st.cache_data(show_spinner=False)
def cached_suggestions(question: str, answer: str) -> list:
    return generate_suggestions(question, answer)


@st.cache_data(show_spinner=False)
def cached_knowledge(question: str, answer: str) -> dict:
    return expand_knowledge(question, answer)


def run_post_answer_tasks(
    question: str,
    answer: str
) -> tuple[list, list, dict]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_suggestions = executor.submit(
            cached_suggestions, question, answer
        )
        future_actions = executor.submit(
            suggest_next_actions, question, answer
        )
        future_knowledge = executor.submit(
            cached_knowledge, question, answer
        )

        suggestions = []
        actions = []
        knowledge = {}

        for future in as_completed([
            future_suggestions,
            future_actions,
            future_knowledge
        ]):
            try:
                result = future.result(timeout=30)
                if future == future_suggestions:
                    suggestions = result
                elif future == future_actions:
                    actions = result
                elif future == future_knowledge:
                    knowledge = result
            except Exception:
                pass

    return suggestions, actions, knowledge


def process_document(uploaded_file) -> bool:
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        is_valid, message = validate_file(tmp_path)
        if not is_valid:
            show_error(message)
            return False

        st.session_state.file_path = tmp_path
        st.session_state.document_name = uploaded_file.name

        tracker = PerformanceTracker()

        tracker.start("reading")
        with st.spinner("📖 Reading document..."):
            text = load_document(tmp_path)
            st.session_state.document_text = text
        tracker.end("reading")

        tracker.start("chunking")
        with st.spinner("✂️ Chunking document..."):
            chunks = chunk_text(text)
        tracker.end("chunking")

        tracker.start("indexing")
        with st.spinner("🔢 Creating search indexes..."):
            create_vector_store(chunks)
            create_bm25_index(chunks)
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
        error_message = handle_error(e, context="document_processing")
        show_error(error_message)
        return False


def handle_question(question: str, answer_mode: str):
    is_valid, validation_message = validate_question(question)
    if not is_valid:
        show_warning(validation_message)
        return

    st.session_state.awaiting_clarification = False
    st.session_state.pending_question = None

    show_chat_message("human", question)
    st.session_state.chat_history.append({
        "role": "human",
        "content": question
    })

    with show_thinking():
        try:
            result = run_workflow(
                question=question,
                memory=st.session_state.memory,
                file_path=st.session_state.file_path,
                answer_mode=answer_mode
            )

            answer = result["answer"]
            evidence = result["evidence"]

            show_chat_message("assistant", answer)
            show_evidence_panel(evidence)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state[f"last_question"] = question
            st.session_state[f"last_answer"] = answer

        except Exception as e:
            error_message = handle_error(e, context="workflow")
            show_error(error_message)
            return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 Get Suggestions", key=f"sug_btn_{len(st.session_state.chat_history)}"):
            with st.spinner("Generating..."):
                suggestions = generate_suggestions(question, answer)
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"- {s}")

    with col2:
        if st.button("🎓 Learn More", key=f"know_btn_{len(st.session_state.chat_history)}"):
            with st.spinner("Expanding knowledge..."):
                knowledge = expand_knowledge(question, answer)
                if knowledge:
                    if knowledge.get("YOUTUBE_SEARCHES"):
                        st.markdown("**📺 YouTube:**")
                        for s in knowledge["YOUTUBE_SEARCHES"]:
                            st.markdown(f"- {s}")
                    if knowledge.get("RELATED_TOPICS"):
                        st.markdown("**🔗 Topics:**")
                        for t in knowledge["RELATED_TOPICS"]:
                            st.markdown(f"- {t}")

    with col3:
        txt_path = export_as_txt(answer, f"answer_{len(st.session_state.chat_history)}")
        with open(txt_path, "rb") as f:
            st.download_button(
                "📥 Download",
                data=f,
                file_name="documind_answer.txt",
                key=f"dl_{len(st.session_state.chat_history)}"
            )


def main():
    apply_custom_css()
    initialize_session()
    show_header()

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
                success = process_document(uploaded_file)
                if success:
                    show_success("✅ Document ready!")
                    st.rerun()

        if st.session_state.document_processed:
            st.markdown("---")
            answer_mode = show_answer_mode_selector()
            st.session_state.answer_mode = answer_mode

            st.markdown("---")
            st.markdown(
                "<p class='sidebar-title'>Current Document</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class='file-info-card'>
                    <p>📄 {st.session_state.document_name}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "<p class='sidebar-title'>Session Stats</p>",
                unsafe_allow_html=True
            )
            show_stats(len(st.session_state.chat_history))

            if len(st.session_state.multi_documents) > 1:
                st.markdown("---")
                st.markdown(
                    "<p class='sidebar-title'>All Documents</p>",
                    unsafe_allow_html=True
                )
                for doc in st.session_state.multi_documents:
                    st.markdown(f"📄 {doc['name']}")

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
                    key=f"qa_{action[:15]}"
                ):
                    with st.spinner(f"Performing: {action}..."):
                        action_result = perform_document_action(
                            action=action,
                            context=st.session_state.document_text[:4000],
                            file_name=st.session_state.document_name
                        )
                        if action_result["success"]:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**{action}**\n\n{action_result['result']}"
                            })
                            st.rerun()

            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.chat_history = []
                st.rerun()

            if st.button("📂 New Document", use_container_width=True):
                st.session_state.document_processed = False
                st.session_state.memory = SessionMemory()
                st.session_state.chat_history = []
                st.session_state.file_path = ""
                st.session_state.document_name = ""
                st.session_state.document_text = ""
                st.session_state.entities = {}
                st.session_state.insights = {}
                st.session_state.document_map = []
                st.session_state.insights_loaded = False
                st.session_state.entities_loaded = False
                st.session_state.map_loaded = False
                st.rerun()

    if not st.session_state.document_processed:
        show_welcome_screen()
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "🔍 Insights",
        "📊 Entities",
        "🗺️ Document Map",
        "✂️ Ask by Selection"
    ])

    with tab1:
        if st.session_state.awaiting_clarification:
            st.markdown(
                """
                <div style='background:rgba(88,166,255,0.05);
                border:1px solid rgba(88,166,255,0.2);
                border-radius:12px;padding:20px;margin-bottom:20px;'>
                <h4 style='color:#58a6ff;'>
                🤔 Let me understand your question better
                </h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            for i, q in enumerate(
                st.session_state.clarification_questions
            ):
                st.markdown(f"**{i+1}.** {q}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Answer anyway", use_container_width=True):
                    q = st.session_state.pending_question
                    st.session_state.awaiting_clarification = False
                    handle_question(q, st.session_state.answer_mode)
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.awaiting_clarification = False
                    st.session_state.pending_question = None
                    st.rerun()

        if len(st.session_state.multi_documents) > 1:
            st.markdown("### 🗂️ Multi-Document Query")
            multi_q = st.text_input(
                "Ask across all uploaded documents:",
                placeholder="Compare the main topics of all documents..."
            )
            if st.button("🔍 Query All Documents"):
                if multi_q:
                    with st.spinner("Querying all documents..."):
                        multi_answer = query_multiple_documents(
                            multi_q,
                            st.session_state.multi_documents
                        )
                        show_chat_message("assistant", multi_answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": multi_answer
                        })
            st.markdown("---")

        for message in st.session_state.chat_history:
            show_chat_message(message["role"], message["content"])

        col_input, col_voice = st.columns([6, 1])

        with col_input:
            question = st.chat_input("Ask anything about your document...")

        with col_voice:
            is_https = (
                st.context.headers.get("x-forwarded-proto") == "https"
                if hasattr(st, "context")
                else False
            )

            if is_https:
                try:
                    from streamlit_mic_recorder import mic_recorder
                    audio = mic_recorder(
                        start_prompt="🎤",
                        stop_prompt="⏹️",
                        key="voice_recorder"
                    )
                    if audio and audio.get("bytes"):
                        with st.spinner("🎤 Transcribing..."):
                            voice_result = transcribe_audio_file(
                                audio["bytes"]
                            )
                            if voice_result["success"]:
                                question = voice_result["text"]
                                show_success(
                                    f"Voice captured: {question}"
                                )
                            else:
                                show_warning(voice_result["error"])
                except Exception as e:
                    logger.error(f"Voice recorder error: {e}")
                    st.button("🎤", help="Voice unavailable")
            else:
                if st.button("🎤", help="Voice works after deployment with HTTPS"):
                    st.session_state["show_voice_tip"] = True

        if st.session_state.get("show_voice_tip"):
            st.info(
                "🎤 Voice input requires HTTPS. "
                "It will work automatically after you deploy to "
                "Streamlit Cloud or any HTTPS server. "
                "For now please type your question."
            )
            if st.button("Got it ✓", key="dismiss_voice_tip"):
                st.session_state["show_voice_tip"] = False
                st.rerun()

    with tab2:
        st.markdown("### 🔍 Living Insight Layer")
        if not st.session_state.insights_loaded:
            if st.button("🧠 Generate Insights", use_container_width=True):
                with st.spinner("Generating insights..."):
                    st.session_state.insights = cached_generate_insights(
                        st.session_state.document_text[:4000]
                    )
                    st.session_state.insights_loaded = True
                    st.rerun()
        else:
            insights = st.session_state.insights
            if insights:
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
                        show_insight_card(title, insights[key], icon)

    with tab3:
        st.markdown("### 📊 Entity Extraction Dashboard")
        if not st.session_state.entities_loaded:
            if st.button(
                "🔍 Extract Entities",
                use_container_width=True
            ):
                with st.spinner("Extracting entities..."):
                    st.session_state.entities = cached_extract_entities(
                        st.session_state.document_text[:3000]
                    )
                    st.session_state.entities_loaded = True
                    st.rerun()
        else:
            entities = st.session_state.entities
            if entities:
                for entity_type, values in entities.items():
                    if values:
                        st.markdown(f"**{entity_type}:**")
                        tags_html = "".join([
                            f"<span style='display:inline-block;background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);border-radius:6px;padding:3px 10px;margin:3px;font-size:0.8rem;color:#d29922;'>{v}</span>"
                            for v in values
                        ])
                        st.markdown(tags_html, unsafe_allow_html=True)
                        st.markdown("")

    with tab4:
        st.markdown("### 🗺️ Document Map")
        if not st.session_state.map_loaded:
            if st.button(
                "🗺️ Generate Document Map",
                use_container_width=True
            ):
                with st.spinner("Building document map..."):
                    st.session_state.document_map = cached_document_map(
                        st.session_state.document_text[:5000]
                    )
                    st.session_state.map_loaded = True
                    st.rerun()
        else:
            doc_map = st.session_state.document_map
            if doc_map:
                for section in doc_map:
                    with st.expander(f"📍 {section['title']}"):
                        st.markdown(
                            section.get("description", "")
                        )
                        if st.button(
                            "Ask about this section",
                            key=f"map_{section['title'][:20]}"
                        ):
                            handle_question(
                                f"Tell me about: {section['title']}",
                                st.session_state.answer_mode
                            )

    with tab5:
        st.markdown("### ✂️ Ask by Selection")
        st.markdown(
            "Paste any specific text from the document and ask about it."
        )

        selected_text = st.text_area(
            "Paste selected text here:",
            placeholder="Paste any paragraph, table, or section...",
            height=150
        )

        action_options = [
            "Explain this",
            "Simplify this",
            "Find issues in this",
            "Summarize this",
            "Translate to simple English",
            "What are the key points here?",
            "Rewrite this professionally"
        ]

        selected_action = st.selectbox(
            "What do you want to do?",
            options=action_options
        )

        if st.button("🔍 Analyze Selection", use_container_width=True):
            if selected_text.strip():
                with st.spinner("Analyzing selected text..."):
                    selection_result = ask_about_selection(
                        selected_text,
                        selected_action
                    )
                    st.markdown("### Result:")
                    st.markdown(selection_result)

                    txt_path = export_as_txt(
                        selection_result, "selection"
                    )
                    with open(txt_path, "rb") as f:
                        st.download_button(
                            "📥 Download Result",
                            data=f,
                            file_name="selection_analysis.txt"
                        )
            else:
                show_warning("Please paste some text first.")


if __name__ == "__main__":
    main()