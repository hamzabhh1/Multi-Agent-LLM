import pyowm
from langgraph.graph import END, StateGraph
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import assistant
import planner
import fn_caller
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "????????"
os.environ["LANGCHAIN_RECURSION_LIMIT"] = "500"
os.environ["COHERE_API_KEY"] = "????????"

class AgentState(TypedDict):
    assistant_history: Annotated[Sequence[BaseMessage],operator.add]
    planner_history: Annotated[Sequence[BaseMessage],operator.add]
    task_executor_history: Annotated[Sequence[BaseMessage],operator.add]
    fn: dict
    error: str
    next_action: str

graph = StateGraph(AgentState)

assistant.link_graph(graph)
planner.link_graph(graph)
fn_caller.link_graph(graph)
assistant.link_conditional_edges(graph)
planner.link_conditional_edges(graph)
fn_caller.link_conditional_edges(graph)

runnable = graph.compile()

runnable.invoke(
    {
        "assistant_history": [],
        "planner_history": [],
        "task_executor_history": [],
        "fn": None,
        "error": None,
        "next_action":" None" 
    }  ,
    config={
        "recursion_limit": 500
    }
)
