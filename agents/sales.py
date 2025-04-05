from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents.base import BaseAgent
from tools.sales import list_inventory

PROMPT = f""" You are a agent responsible for the sales department of a
        car dealership:

        1. Choose your action using the tools that are available to you.
        2. If there is no tool to call with the user request,
            ask for more context about what the user want.
        3. Return the response to the Supervisor agent

        Current Date: {datetime.now()}
        """


@tool
class ToSales(BaseModel):
    """Call this tool when the user want to buy a car"""


def get_agent(llm: BaseChatModel) -> CompiledStateGraph:
    agent = BaseAgent(llm=llm, prompt_content=PROMPT, tools=[list_inventory])
    return agent.build_agent()
