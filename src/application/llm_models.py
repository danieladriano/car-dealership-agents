from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


class SupportedLLMs(Enum):
    llama3_1 = "llama3.1:8b"
    llama3_2 = "llama3.2"
    mistral_7b = "mistral:7b"
    qwen2_5_14b = "qwen2.5:14b"
    gemini2_0_flash = "gemini-2.0-flash"


def get_llm(llm_model: SupportedLLMs) -> BaseChatModel:
    if llm_model in SupportedLLMs:
        if llm_model == SupportedLLMs.gemini2_0_flash:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=1.5,
                top_p=0.95,
                max_retries=2,
            )
        return ChatOllama(model=llm_model.value)
    raise Exception("LLM not supported")
