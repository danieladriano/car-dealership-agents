from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


class SupportedLLMs(Enum):
    llama3_1 = "llama3.1"
    llama3_2 = "llama3.2"
    mistral_7b = "mistral_7b"


def get_llm(llm_model: SupportedLLMs) -> BaseChatModel:
    if llm_model == SupportedLLMs.llama3_1:
        return ChatOllama(model="llama3.1:8b")
    if llm_model == SupportedLLMs.llama3_2:
        return ChatOllama(model="llama3.2")
    if llm_model == SupportedLLMs.mistral_7b:
        return ChatOllama(model="mistral:7b")
    raise Exception("LLM not supported")
