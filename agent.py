import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from assets import CancelTestDriveMessages
from graph_memory import GraphMemory
from tools.sales import list_inventory, inventory_information
from tools.test_drive import (
    CancelTestDrive,
    cancel_test_drive,
    list_test_drives,
    schedule_test_drive,
)

logger = logging.getLogger("ai-chat")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class CarModelDetails(BaseModel):
    """
    If the user wants more tecnical details about a specific car model.
    """

    user_request: str = Field(description="The user request about the car model")


class Agent:
    def __init__(self, llm: BaseChatModel, data_path: Path, cognee_path: Path) -> None:
        self._llm = llm
        self._runnable = self._get_prompt_template() | self._llm.bind_tools(
            [
                list_inventory,
                inventory_information,
                list_test_drives,
                schedule_test_drive,
                CancelTestDrive,
                CarModelDetails,
            ]
        )
        self._graph_memory = GraphMemory(data_path=data_path, cognee_path=cognee_path)

    def _get_prompt_template(self) -> ChatPromptTemplate:
        content = f""" You are a helpfull Volkswagen Dealership Assistant.
                    Your main objective is to help the user to find a perfect car for his needs.
                    Be polite and helpfull.

                    Current Date: {datetime.now()}
                    """
        return ChatPromptTemplate(
            [
                SystemMessage(content=content),
                MessagesPlaceholder(variable_name="conversation"),
            ]
        )

    def call_model(self, state: State) -> State:
        logger.info("Calling model")
        response = self._runnable.invoke({"conversation": state["messages"]})

        return {"messages": [response]}  # type: ignore

    def cancel_test_drive_node(self, state: State) -> State:
        if isinstance(state["messages"][-1], AIMessage):
            tool_call = state["messages"][-1].tool_calls[0]

        cancel = CancelTestDrive.model_validate(tool_call["args"])
        user_answer = interrupt(CancelTestDriveMessages.CONFIRM.format(cancel.code))

        content = CancelTestDriveMessages.NOT_CANCEL
        if user_answer.content == "y":
            content = CancelTestDriveMessages.ERROR_CANCEL
            if cancel_test_drive(code=cancel.code):
                content = CancelTestDriveMessages.CANCELD

        return {
            "messages": [
                ToolMessage(content=content, tool_call_id=tool_call["id"], type="tool")
            ]
        }

    async def car_model_details_node(self, state: State) -> State:
        if isinstance(state["messages"][-1], AIMessage):
            tool_call = state["messages"][-1].tool_calls[0]

        response = await self._graph_memory.search(
            query_text=tool_call["args"]["user_request"]
        )
        return {
            "messages": [
                ToolMessage(content=response, tool_call_id=tool_call["id"], type="tool")
            ]
        }

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info(
                f"ToolCall - {last_message.tool_calls[0]['name']} - Args {last_message.tool_calls[0]['args']}"
            )
            if last_message.tool_calls[0]["name"] == "CancelTestDrive":
                return "cancel_test_drive"
            if last_message.tool_calls[0]["name"] == "CarModelDetails":
                return "car_model_details"
            return "tools"
        return END

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
                    inventory_information,
                    list_test_drives,
                    schedule_test_drive,
                ]
            ),
        )
        graph_builder.add_node(
            node="cancel_test_drive", action=self.cancel_test_drive_node
        )
        graph_builder.add_node(
            node="car_model_details", action=self.car_model_details_node
        )

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model",
            path=self.conditional_router,
            path_map=["tools", "cancel_test_drive", "car_model_details", END],
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")
        graph_builder.add_edge(start_key="cancel_test_drive", end_key="call_model")
        graph_builder.add_edge(start_key="car_model_details", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)
