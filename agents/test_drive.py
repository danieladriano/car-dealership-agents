from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents.base import BaseAgent
from tools.test_drive import list_test_drives, schedule_test_drive

PROMPT = f""" You are a agent responsible for the test drive department of a
        car dealership:

        1. Choose your action using the tools that are available to you.
        2. If there is no tool to call with the user request,
            ask for more context about what the user want.
        3. Always elaborate a complete response to the user.
        4. Never reference our tools to the user

        Current Date: {datetime.now()}
        """


@tool
class ToTestDrive(BaseModel):
    """Call this tool when the user want to talk about test driver"""


def get_agent(llm: BaseChatModel) -> CompiledStateGraph:
    agent = BaseAgent(
        llm=llm, prompt_content=PROMPT, tools=[schedule_test_drive, list_test_drives]
    )
    return agent.build_agent()
