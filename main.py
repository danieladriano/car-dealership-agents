import uuid
from datetime import datetime
from typing import Annotated, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from llm_models import SupportedLLMs, get_llm
from tools.sales import list_inventory
from tools.test_drive import cancel_test_drive, list_test_drives, schedule_test_drive


class CancelTestDrive(BaseModel):
    """Cancel a test drive"""

    code: int = Field(description="The code of the test drive to cancel")


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
                    - CancelTestDrive

                    Current Date: {datetime.now()}
                    """
        system_message = SystemMessage(content=content)
        return RunnableCallable(lambda state: [system_message] + state, name="Prompt")

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"[conditional_router] {last_message.tool_calls[0]['name']}")
            if last_message.tool_calls[0]["name"] == "CancelTestDrive":
                return "cancel_test_drive"
            return "tools"
        return END

    def call_model(self, state: State) -> State:
        assistant_runnable = self._prompt | self._llm.bind_tools(
            [list_inventory, list_test_drives, schedule_test_drive, CancelTestDrive]
        )
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}  # type: ignore

    def cancel_test_drive_node(self, state: State) -> State:
        tool_call = state["messages"][-1].tool_calls[0]
        cancel = CancelTestDrive.model_validate(tool_call["args"])
        user_answer = interrupt(
            f"Do you confirm the cancel of test drive code {cancel.code}? [y/n]"
        )
        content = "User gave up canceling, He want to do the test drive."
        if user_answer.content == "y":
            content = (
                "Error when canceling the test drive. Need to call do the dealership."
            )
            if cancel_test_drive(code=cancel.code):
                content = "Test drive canceled."
        return {
            "messages": [
                ToolMessage(content=content, tool_call_id=tool_call["id"], type="tool")
            ]
        }

    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode([list_inventory, list_test_drives, schedule_test_drive]),
        )
        graph_builder.add_node("cancel_test_drive", self.cancel_test_drive_node)

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model",
            path=self.conditional_router,
            path_map=["tools", "cancel_test_drive", END],
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")
        graph_builder.add_edge(start_key="cancel_test_drive", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)


def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> None:
    state = graph.get_state(config=config)
    message = {"messages": [("user", user_input)]}
    if state.tasks and state.tasks[0].interrupts:
        message = Command(resume=HumanMessage(content=user_input))

    events = graph.invoke(input=message, config=config, stream_mode="values")

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
