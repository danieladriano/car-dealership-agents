from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from application.graph_memory import GraphMemory
from src.application.agents.dealership import DealershipAgent

def build_agent(llm: BaseChatModel, graph_memory: GraphMemory) -> CompiledStateGraph:
    checkpointer = MemorySaver()
    chatbot = DealershipAgent(llm=llm, graph_memory=graph_memory)
    return chatbot.build_agent(checkpointer=checkpointer)
