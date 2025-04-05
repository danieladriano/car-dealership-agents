import logging
from datetime import datetime
from typing import Annotated, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import TypedDict

from tools.sales import list_inventory
from tools.test_drive import (
    CancelTestDrive,
    cancel_test_drive,
    list_test_drives,
    schedule_test_drive,
)

logger = logging.getLogger("ai-chat")


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
            logger.info(
                f"ToolCall - {last_message.tool_calls[0]['name']} - Args {last_message.tool_calls[0]['args']}"
            )
            if last_message.tool_calls[0]["name"] == "CancelTestDrive":
                return "cancel_test_drive"
            return "tools"
        return END

    def call_model(self, state: State) -> State:
        logger.info("Calling model")
        assistant_runnable = self._prompt | self._llm.bind_tools(
            [list_inventory, list_test_drives, schedule_test_drive, CancelTestDrive]
        )
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}  # type: ignore

    def cancel_test_drive_node(self, state: State) -> State:
        if isinstance(state["messages"][-1], AIMessage):
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
