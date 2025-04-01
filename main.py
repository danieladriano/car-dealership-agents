import uuid
from datetime import datetime
from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import TypedDict

from llm_models import SupportedLLMs, get_llm
from tools.sales import list_inventory


class State(TypedDict):
    messages: Annotated[list, add_messages]


class Agent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    @property
    def _prompt(self) -> RunnableCallable:
        content = f""" You are a helpfull Volkswagen Dealership Assistant

                    You need to follow this rules:
                    1. Choose your action using the tools that are available to you.
                    2. If there is no tool to call with the user request,
                       ask for more context about what the user want.
                    3. Always elaborate a complete response to the user.
                    4. Never reference our tools to the user

                    Current Date: {datetime.now()}
                    """
        system_message = SystemMessage(content=content)
        return RunnableCallable(lambda state: [system_message] + state, name="Prompt")

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(self, state: State) -> State:
        assistant_runnable = self._prompt | self._llm.bind_tools([list_inventory])
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}

    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode([list_inventory]),
        )

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model", path=self.conditional_router, path_map=["tools", END]
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)


def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> None:
    messages = {"messages": [("user", user_input)]}
    events = graph.invoke(input=messages, config=config, stream_mode="values")
    events["messages"][-1].pretty_print()


def main() -> None:
    llm = get_llm(llm_model=SupportedLLMs.llama3_1)
    checkpointer = MemorySaver()
    chatbot = Agent(llm=llm)
    graph = chatbot.build_agent(checkpointer=checkpointer)

    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    print("Assistant: Welcome! How can I help you today?")
    while True:
        try:
            print(80*"=")
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph=graph, config=config, user_input=user_input)
        except Exception as e:
            print(f"Error {e}")


if __name__ == "__main__":
    main()
