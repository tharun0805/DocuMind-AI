from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.intent_agent import classify_intent
from agents.planner_agent import plan_route
from agents.retriever_agent import retrieve_context
from agents.qa_agent import generate_answer
from loguru import logger


class AgentState(TypedDict):
    question: str
    intent: str
    route: str
    context: str
    answer: str


def intent_node(state: AgentState) -> AgentState:
    logger.info("Running intent node...")
    state["intent"] = classify_intent(state["question"])
    return state


def planner_node(state: AgentState) -> AgentState:
    logger.info("Running planner node...")
    state["route"] = plan_route(state["intent"])
    return state


def retriever_node(state: AgentState) -> AgentState:
    logger.info("Running retriever node...")
    state["context"] = retrieve_context(state["question"])
    return state


def qa_node(state: AgentState) -> AgentState:
    logger.info("Running QA node...")
    state["answer"] = generate_answer(state["question"], state["context"])
    return state


def route_decision(state: AgentState) -> str:
    if state["route"] == "dataframe":
        return "dataframe"
    return "retriever"


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent", intent_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("qa", qa_node)

    workflow.set_entry_point("intent")

    workflow.add_edge("intent", "planner")

    workflow.add_conditional_edges(
        "planner",
        route_decision,
        {
            "retriever": "retriever",
            "dataframe": "retriever"
        }
    )

    workflow.add_edge("retriever", "qa")
    workflow.add_edge("qa", END)

    return workflow.compile()


def run_workflow(question: str) -> str:
    logger.info(f"Running workflow for question: {question}")

    app = build_workflow()

    initial_state = AgentState(
        question=question,
        intent="",
        route="",
        context="",
        answer=""
    )

    final_state = app.invoke(initial_state)
    return final_state["answer"]