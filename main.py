import uuid
from datetime import datetime
from typing import Annotated, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import TypedDict

from llm_models import SupportedLLMs, get_llm
from tools.sales import list_inventory
from tools.test_drive import cancel_test_drive, list_test_drives, schedule_test_drive


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class Agent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    @property
    def _prompt(self) -> RunnableCallable:
        content = f""" You are a helpfull Volkswagen Dealership Assistant

                    You must use the tools available to deal with the user request.

                    This is the tools available to you:

                    - list_inventory
                    - list_test_drivers
                    - schedule_test_drive
                    - cancel_test_drive

                    Current Date: {datetime.now()}
                    """
        system_message = SystemMessage(content=content)
        return RunnableCallable(lambda state: [system_message] + state, name="Prompt")

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"[conditional_router] {last_message.tool_calls[0]['name']}")
            return "tools"
        return END

    def call_model(self, state: State) -> State:
        assistant_runnable = self._prompt | self._llm.bind_tools(
            [list_inventory, list_test_drives, schedule_test_drive, cancel_test_drive]
        )
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}  # type: ignore

    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode(
                [
                    list_inventory,
                    list_test_drives,
                    schedule_test_drive,
                    cancel_test_drive,
                ]
            ),
        )

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model",
            path=self.conditional_router,
            path_map=["tools", END],
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)


def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> None:
    state = graph.get_state(config=config)
    if state.tasks and state.tasks[0].interrupts:
        graph_input = Command(resume=HumanMessage(content=user_input))
    else:
        graph_input = ("user", user_input)

    messages = {"messages": [graph_input]}
    events = graph.invoke(input=messages, config=config, stream_mode="values")

    state = graph.get_state(config=config)
    if state.tasks and state.tasks[0].interrupts:
        print(state.tasks[0].interrupts[0].value)
    else:
        events["messages"][-1].pretty_print()


def main() -> None:
    llm = get_llm(llm_model=SupportedLLMs.qwen2_5_14b)
    checkpointer = MemorySaver()
    chatbot = Agent(llm=llm)
    graph = chatbot.build_agent(checkpointer=checkpointer)

    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    print("Assistant: Welcome! How can I help you today?")
    while True:
        try:
            print(80 * "=")
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph=graph, config=config, user_input=user_input)
        except Exception as e:
            print(f"Error {e}")


if __name__ == "__main__":
    main()
