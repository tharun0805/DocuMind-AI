import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import tempfile
import streamlit as st
from loguru import logger

from ingestion.document_loader import load_document
from chunking.text_chunker import chunk_text
from vector_store.faiss_store import create_vector_store
from vector_store.bm25_store import create_bm25_index
from graph.workflow import run_workflow
from memory.session_memory import SessionMemory
from app.ui_components import (
    apply_custom_css,
    show_header,
    show_welcome_screen,
    show_chat_message,
    show_success,
    show_error,
    show_file_info,
    show_stats,
    show_thinking
)


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session():
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "file_path" not in st.session_state:
        st.session_state.file_path = ""
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = ""


def process_document(uploaded_file) -> bool:
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        st.session_state.file_path = tmp_path
        st.session_state.document_name = uploaded_file.name

        with st.spinner("📖 Reading document..."):
            text = load_document(tmp_path)

        with st.spinner("✂️ Chunking document..."):
            chunks = chunk_text(text)

        with st.spinner("🔢 Creating embeddings and indexes..."):
            create_vector_store(chunks)
            create_bm25_index(chunks)

        st.session_state.document_processed = True
        st.session_state.memory = SessionMemory()
        st.session_state.chat_history = []

        logger.info(f"Document processed: {uploaded_file.name}")
        return True

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        show_error(f"Error processing document: {str(e)}")
        return False


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
            help="Upload any document to start asking questions",
            label_visibility="collapsed"
        )

        if uploaded_file:
            show_file_info(uploaded_file.name, uploaded_file.size)

            if st.button(
                "🚀 Process Document",
                use_container_width=True,
                type="primary"
            ):
                success = process_document(uploaded_file)
                if success:
                    show_success("✅ Document ready!")
                    st.rerun()

        if st.session_state.document_processed:
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

            st.markdown("---")

            if st.button(
                "🗑️ Clear Conversation",
                use_container_width=True
            ):
                st.session_state.memory = SessionMemory()
                st.session_state.chat_history = []
                st.rerun()

            if st.button(
                "📂 Upload New Document",
                use_container_width=True
            ):
                st.session_state.document_processed = False
                st.session_state.memory = SessionMemory()
                st.session_state.chat_history = []
                st.session_state.file_path = ""
                st.session_state.document_name = ""
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<p class='sidebar-title'>Supported Formats</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style='color: #8b949e; font-size: 0.8rem; line-height: 2;'>
                📕 PDF Documents<br>
                📘 Word Documents<br>
                📗 Excel Spreadsheets<br>
                📙 PowerPoint Files<br>
                📊 CSV Data Files<br>
                📝 Text Files
            </div>
            """,
            unsafe_allow_html=True
        )

    if not st.session_state.document_processed:
        show_welcome_screen()
        return

    for message in st.session_state.chat_history:
        show_chat_message(message["role"], message["content"])

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:
        show_chat_message("human", question)
        st.session_state.chat_history.append({
            "role": "human",
            "content": question
        })

        with show_thinking():
            try:
                answer = run_workflow(
                    question=question,
                    memory=st.session_state.memory,
                    file_path=st.session_state.file_path
                )

                show_chat_message("assistant", answer)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                show_error(error_msg)
                logger.error(error_msg)


if __name__ == "__main__":
    main()