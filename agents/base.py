from typing import Annotated, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class BaseAgent:
    def __init__(
        self, llm: BaseChatModel, prompt_content: str, tools: List[BaseTool]
    ) -> None:
        self._llm = llm
        self._prompt_content = prompt_content
        self._tools = tools

    @property
    def _prompt(self) -> RunnableCallable:
        system_message = SystemMessage(content=self._prompt_content)
        return RunnableCallable(lambda state: [system_message] + state, name="Prompt")

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"[SupervisorAgent][conditional_router] {last_message.tool_calls[-1]['name']}")
            return "tools"
        return END

    def call_model(self, state: State) -> State:
        assistant_runnable = self._prompt | self._llm.bind_tools(self._tools)
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}  # type: ignore

    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode(self._tools),
        )

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model", path=self.conditional_router, path_map=["tools", END]
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)
