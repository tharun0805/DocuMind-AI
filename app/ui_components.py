import streamlit as st
from loguru import logger


def apply_custom_css():
    st.markdown(
        """
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
            padding: 40px 20px 20px 20px;
            background: linear-gradient(135deg, rgba(88,166,255,0.05) 0%, rgba(63,185,80,0.05) 100%);
            border-radius: 16px;
            border: 1px solid rgba(88,166,255,0.1);
            margin-bottom: 30px;
        }
        .main-header h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #58a6ff 0%, #3fb950 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        .main-header p { color: #8b949e; font-size: 1.1rem; font-weight: 300; }
        .sidebar-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: #8b949e !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 15px rgba(88,166,255,0.3) !important;
        }
        .file-info-card {
            background: linear-gradient(135deg, rgba(63,185,80,0.08) 0%, rgba(63,185,80,0.03) 100%);
            border: 1px solid rgba(63,185,80,0.2);
            border-radius: 10px;
            padding: 12px 16px;
            margin: 10px 0;
        }
        .file-info-card p { color: #3fb950; font-size: 0.85rem; font-weight: 500; margin: 0; }
        .stats-card {
            background: rgba(88,166,255,0.05);
            border: 1px solid rgba(88,166,255,0.1);
            border-radius: 10px;
            padding: 12px 16px;
            margin: 8px 0;
            text-align: center;
        }
        .stats-card h3 { color: #58a6ff; font-size: 1.5rem; font-weight: 700; margin: 0; }
        .stats-card p { color: #8b949e; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
        .insight-card {
            background: rgba(63,185,80,0.05);
            border: 1px solid rgba(63,185,80,0.15);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .insight-card h4 { color: #3fb950; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 30px;
            text-align: left;
        }
        .feature-card {
            background: rgba(22,27,34,0.8);
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            border-color: #58a6ff;
            background: rgba(88,166,255,0.05);
            transform: translateY(-2px);
        }
        .feature-card .icon { font-size: 1.8rem; margin-bottom: 10px; }
        .feature-card h4 { color: #e6edf3; font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; }
        .feature-card p { color: #8b949e; font-size: 0.8rem; margin: 0; line-height: 1.5; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
        hr { border-color: #21262d !important; }
        </style>
        """,
        unsafe_allow_html=True
    )


def show_header():
    st.markdown(
        """
        <div class='main-header'>
            <h1>🧠 DocuMind AI</h1>
            <p>Intelligent Document Intelligence Platform — Powered by Gemini and LangGraph</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_welcome_screen():
    st.markdown(
        """
        <div style='text-align:center;margin-top:80px;padding:60px 40px;'>
            <h2 style='color:#e6edf3;font-size:2rem;font-weight:600;'>
                Upload a document to get started
            </h2>
            <p style='color:#8b949e;'>
                Ask questions, get insights, and analyse your documents using AI
            </p>
            <div class='feature-grid'>
                <div class='feature-card'>
                    <div class='icon'>📄</div>
                    <h4>Multi-Format Support</h4>
                    <p>PDF, Word, Excel, PowerPoint, CSV and TXT files supported</p>
                </div>
                <div class='feature-card'>
                    <div class='icon'>🔍</div>
                    <h4>Hybrid Search</h4>
                    <p>FAISS semantic search combined with BM25 keyword search</p>
                </div>
                <div class='feature-card'>
                    <div class='icon'>🤖</div>
                    <h4>Multi-Agent AI</h4>
                    <p>Specialized agents for intent, planning, retrieval and answers</p>
                </div>
                <div class='feature-card'>
                    <div class='icon'>💬</div>
                    <h4>Session Memory</h4>
                    <p>Ask follow up questions naturally with full conversation memory</p>
                </div>
                <div class='feature-card'>
                    <div class='icon'>📊</div>
                    <h4>Data Analysis</h4>
                    <p>Ask calculation questions on Excel and CSV files</p>
                </div>
                <div class='feature-card'>
                    <div class='icon'>🔒</div>
                    <h4>100% Private</h4>
                    <p>Your documents never leave your machine</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_chat_message(role: str, content: str):
    if role == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(content)


def show_answer_mode_selector() -> str:
    st.markdown(
        "<p class='sidebar-title'>Answer Mode</p>",
        unsafe_allow_html=True
    )
    mode = st.selectbox(
        "Select mode",
        options=["detailed", "quick", "bullet", "beginner", "executive", "table"],
        format_func=lambda x: {
            "detailed": "📝 Detailed Explanation",
            "quick": "⚡ Quick Answer",
            "bullet": "• Bullet Points",
            "beginner": "🎓 Beginner Friendly",
            "executive": "💼 Executive Summary",
            "table": "📊 Table Format"
        }[x],
        label_visibility="collapsed"
    )
    return mode


def show_evidence_panel(evidence: list):
    if not evidence:
        return
    with st.expander("📎 View Evidence — Sources Used"):
        for i, chunk in enumerate(evidence, 1):
            st.markdown(
                f"""
                <div style='background:rgba(88,166,255,0.05);
                border-left:3px solid #58a6ff;
                border-radius:0 8px 8px 0;
                padding:12px 16px;margin-bottom:12px;
                font-size:0.85rem;color:#8b949e;line-height:1.6;'>
                <strong style='color:#58a6ff;'>Source {i}:</strong><br>{chunk}
                </div>
                """,
                unsafe_allow_html=True
            )


def show_file_info(file_name: str, file_size: int):
    size_kb = file_size / 1024
    st.markdown(
        f"""
        <div class='file-info-card'>
            <p>📄 <strong>{file_name}</strong> — {size_kb:.1f} KB</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_stats(message_count: int):
    st.markdown(
        f"""
        <div class='stats-card'>
            <h3>{message_count}</h3>
            <p>Messages</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_insight_card(title: str, content: str, icon: str = "💡"):
    st.markdown(
        f"""
        <div class='insight-card'>
            <h4>{icon} {title}</h4>
            <p style='color:#8b949e;font-size:0.85rem;line-height:1.6;'>{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_success(message: str):
    st.success(message)
    logger.info(f"UI success: {message}")


def show_error(message: str):
    st.error(message)
    logger.error(f"UI error: {message}")


def show_warning(message: str):
    st.warning(message)
    logger.warning(f"UI warning: {message}")


def show_thinking():
    return st.spinner("🧠 DocuMind is analysing your question...")