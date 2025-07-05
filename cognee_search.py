import asyncio
import logging
import os
from pathlib import Path

import cognee
import streamlit as st

logger = logging.getLogger(__name__)


def load_cognee() -> None:
    data_path = str(
        Path(  # noqa: F821
            os.path.join(Path(__file__).parent, ".data_storage")
        ).resolve()
    )

    cognee_path = str(
        Path(os.path.join(Path(__file__).parent, ".cognee_system")).resolve()
    )

    cognee.config.data_root_directory(data_root_directory=data_path)
    cognee.config.system_root_directory(system_root_directory=cognee_path)


async def search(query_text: str) -> str:
    logger.info(f"Calling cognee: {query_text}")
    response = await cognee.search(
        query_text=query_text, query_type=cognee.SearchType.RAG_COMPLETION
    )
    logger.info(f"Response: {response}")
    return response[0]


st.title("Simple chat")

load_cognee()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(body=prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(search(query_text=prompt))
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
