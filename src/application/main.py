import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from application.llm_models import SupportedLLMs, get_llm
from src.application.agents.factory import build_agent
from src.application.graph_memory import GraphMemory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


async def invoke_graph(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> str:
    message = _build_graph_input(graph=graph, config=config, user_input=user_input)
    events = await graph.ainvoke(input=message, config=config, stream_mode="values")

    interrupt = _get_interrupt(graph=graph, config=config)
    if interrupt:
        return interrupt.value

    return events["messages"][-1].content


st.title("VW - Car Dealership")
if "graph" not in st.session_state:
    logger.info("Loading LLM and Graph")

    llm = get_llm(llm_model=SupportedLLMs.gemini2_0_flash)

    root_path = Path(__file__).parent.parent.parent
    data_path = Path(root_path, ".data_storage").resolve()
    cognee_path = Path(root_path, ".cognee_system").resolve()
    graph_memory = GraphMemory(data_path=data_path, cognee_path=cognee_path)

    graph = build_agent(llm=llm, graph_memory=graph_memory)
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    st.session_state.graph = graph
    st.session_state.config = config
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(body=prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(
            invoke_graph(
                graph=st.session_state.graph,
                config=st.session_state.config,
                user_input=prompt,
            )
        )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
