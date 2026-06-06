from typing import TypedDict
import streamlit as st
from langgraph.graph import StateGraph, END
from loguru import logger


class AgentState(TypedDict):
    question:     str
    intent:       str
    route:        str
    context:      str
    answer:       str
    chat_history: str
    file_path:    str
    doc_text:     str
    evidence:     list
    answer_mode:  str


def _intent_node(state: AgentState) -> AgentState:
    from agents.intent_agent import classify_intent
    state["intent"] = classify_intent(state["question"])
    return state


def _planner_node(state: AgentState) -> AgentState:
    from agents.planner_agent import plan_route
    state["route"] = plan_route(state["intent"], state.get("file_path", ""))
    return state


def _retriever_node(state: AgentState) -> AgentState:
    from agents.retriever_agent import retrieve_context
    state["context"] = retrieve_context(state["question"])
    return state


def _qa_node(state: AgentState) -> AgentState:
    from agents.qa_agent import generate_answer
    result = generate_answer(
        question=state["question"],
        context=state["context"],
        chat_history=state["chat_history"],
        answer_mode=state.get("answer_mode", "detailed"),
        doc_text=state.get("doc_text", ""),
    )
    state["answer"]   = result["answer"]
    state["evidence"] = result["evidence"]
    return state


def _dataframe_node(state: AgentState) -> AgentState:
    from agents.dataframe_agent import run_dataframe_agent
    try:
        state["answer"] = run_dataframe_agent(
            question=state["question"],
            file_path=state["file_path"],
        )
    except Exception as e:
        logger.error(f"Dataframe node error: {e}")
        # Fallback to retrieval
        from agents.retriever_agent import retrieve_context
        from agents.qa_agent import generate_answer
        ctx    = retrieve_context(state["question"])
        result = generate_answer(
            question=state["question"], context=ctx,
            chat_history=state["chat_history"],
            answer_mode=state.get("answer_mode", "detailed"),
            doc_text=state.get("doc_text", ""),
        )
        state["answer"]   = result["answer"]
        state["evidence"] = result.get("evidence", [])
    state.setdefault("evidence", [])
    return state


def _route_decision(state: AgentState) -> str:
    return "dataframe" if state["route"] == "dataframe" else "retriever"


def _compiled_app():
    """Compiled fresh per session — always picks up latest agent code."""
    logger.debug("Compiling LangGraph workflow...")
    wf = StateGraph(AgentState)
    wf.add_node("intent",    _intent_node)
    wf.add_node("planner",   _planner_node)
    wf.add_node("retriever", _retriever_node)
    wf.add_node("qa",        _qa_node)
    wf.add_node("dataframe", _dataframe_node)
    wf.set_entry_point("intent")
    wf.add_edge("intent", "planner")
    wf.add_conditional_edges(
        "planner", _route_decision,
        {"retriever": "retriever", "dataframe": "dataframe"},
    )
    wf.add_edge("retriever", "qa")
    wf.add_edge("dataframe", END)
    wf.add_edge("qa", END)
    return wf.compile()


@st.cache_resource(show_spinner=False)
def _get_app(_sid: str = "default"):
    """One compiled graph per Streamlit session."""
    return _compiled_app()


def run_workflow(
    question: str,
    memory=None,
    file_path: str = "",
    answer_mode: str = "detailed",
    doc_text: str = "",
) -> dict:
    logger.debug(f"Workflow: {question[:60]}")
    app = _get_app()

    chat_history = ""
    if memory and not memory.is_empty():
        chat_history = memory.get_history_as_text()

    initial = AgentState(
        question=question, intent="", route="", context="", answer="",
        chat_history=chat_history, file_path=file_path, doc_text=doc_text,
        evidence=[], answer_mode=answer_mode,
    )

    try:
        final = app.invoke(initial)
    except Exception as e:
        logger.error(f"Workflow invoke error: {e}")
        return {"answer": "", "evidence": []}

    answer   = final.get("answer", "") or ""
    evidence = final.get("evidence", []) or []

    if answer.strip() and memory is not None:
        memory.add_human_message(question)
        memory.add_ai_message(answer)

    return {"answer": answer, "evidence": evidence}