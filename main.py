import logging
import uuid
from typing import Optional

from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from agent import Agent
from llm_models import SupportedLLMs, get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-chat")


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


def main() -> None:
    llm = get_llm(llm_model=SupportedLLMs.qwen2_5_14b)
    checkpointer = MemorySaver()
    chatbot = Agent(llm=llm)
    graph = chatbot.build_agent(checkpointer=checkpointer)

    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    logger.info("Welcome! How can I help you today?")
    while True:
        try:
            logger.info(80 * "=")
            user_input = input("INFO:ai-chat:User: ")
            logger.info(80 * "-")

            if user_input.lower() in ["quit", "exit"]:
                logger.info("Goodbye!")
                break

            ai_message = stream_graph_updates(
                graph=graph, config=config, user_input=user_input
            )
            logger.info(80 * "-")
            logger.info(ai_message)
        except Exception as ex:
            logger.error(ex)


if __name__ == "__main__":
    main()
