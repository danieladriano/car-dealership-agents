import streamlit as st
import logging
from llm_models import SupportedLLMs, get_llm
from agent import Agent
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
import uuid

from typing import Optional

from langchain_core.messages.human import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

logger = logging.getLogger(__name__)


logger.info("Loading LLM and Graph")
llm = get_llm(llm_model=SupportedLLMs.gemini2_0_flash)
checkpointer = MemorySaver()
chatbot = Agent(llm=llm)
graph = chatbot.build_agent(checkpointer=checkpointer)

config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})


def _get_interrupt(
    graph: CompiledStateGraph, config: RunnableConfig
) -> Optional[Interrupt]:
    state = graph.get_state(config=config)
    if state.tasks and state.tasks[0].interrupts:
        return state.tasks[0].interrupts[0]
    return None


def _build_graph_input(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> dict[str, list[tuple[str, str]]] | Command:
    interrupt = _get_interrupt(graph=graph, config=config)
    if interrupt:
        return Command(resume=HumanMessage(content=user_input))
    return {"messages": [("user", user_input)]}


def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> str:
    message = _build_graph_input(graph=graph, config=config, user_input=user_input)
    events = graph.invoke(input=message, config=config, stream_mode="values")

    interrupt = _get_interrupt(graph=graph, config=config)
    if interrupt:
        return interrupt.value

    return events["messages"][-1].content


st.title("VW - Car Dealership")

logger.info(f"Session state {st.session_state}")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(body=prompt)

    with st.chat_message("assistant"):
        response = stream_graph_updates(graph=graph, config=config, user_input=prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
