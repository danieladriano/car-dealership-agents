from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base import BaseAgent, State
from agents.sales import ToSales
from agents.sales import get_agent as get_sales_agent
from agents.test_drive import ToTestDrive
from agents.test_drive import get_agent as get_test_drive_agent

PROMPT = f""" You are a helpfull Volkswagen Dealership Assistant

        You need to follow this rules:
        1. Choose your action using the tools that are available to you.
        2. If there is no tool to call with the user request,
            ask for more context about what the user want.
        3. Always elaborate a complete response to the user considering the ansewers from the other agents.
        4. Never reference our tools to the user

        Current Date: {datetime.now()}
        """


class SupervisorAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm, PROMPT, tools=[ToSales, ToTestDrive])

    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"[SupervisorAgent][conditional_router] {last_message.tool_calls[-1]['name']}")
            if last_message.tool_calls[-1]["name"] == ToSales.name:
                return "sales"
            if last_message.tool_calls[-1]["name"] == ToTestDrive.name:
                return "test_drive"
        print(f"[SupervisorAgent][conditional_router] {messages}")
        return END

    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(node="sales", action=get_sales_agent(llm=self._llm))  # type: ignore
        graph_builder.add_node(
            node="test_drive", action=get_test_drive_agent(llm=self._llm)
        )  # type: ignore

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model",
            path=self.conditional_router,
            path_map=["sales", "test_drive", END],
        )
        graph_builder.add_edge(start_key="sales", end_key="call_model")
        graph_builder.add_edge(start_key="test_drive", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)


def get_agent(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    agent = SupervisorAgent(llm=llm)
    return agent.build_agent(checkpointer=checkpointer)
