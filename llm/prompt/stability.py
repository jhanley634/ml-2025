#! /usr/bin/env uv run streamlit run --server.runOnSave true --server.headless true --browser.gatherUsageStats false

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import streamlit as st
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from streamlit.delta_generator import DeltaGenerator

from llm.prompt.config import CHAT_PROMPT_TEMPLATE


@dataclass
class LLM:
    name: str
    model_instance: OllamaLLM


def _get_model(name: str) -> LLM:
    return LLM(name, OllamaLLM(model=name))


models = [
    _get_model("phi4"),
    _get_model("gemma3:12b"),
    # _get_model("mixtral"),
]


def _msg(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
    }


@beartype
async def get_streaming_response_from_model(
    prompt_template: ChatPromptTemplate,
    model: LLM,
    user_input: str,
) -> AsyncGenerator[str]:
    chain = prompt_template | model.model_instance
    yield f"**{model.name}**:\n\n"
    response = await asyncio.to_thread(chain.astream, {"question": user_input})
    async for token in response:
        yield token


@beartype
async def handle_model_responses(
    prompt: ChatPromptTemplate,
    model: LLM,
    user_input: str,
    container: DeltaGenerator,
) -> None:
    async for token in get_streaming_response_from_model(prompt, model, user_input):
        msg = {"role": f"{model.name} ai", "content": token}
        st.session_state.messages.append(msg)
        text = "".join(elt["content"] for elt in st.session_state.messages)
        container.write(f"**{model.name}**:\n\n{text}")


def response_from_multiple_models() -> None:
    st.markdown(
        "<h2 style='text-align: center; font-family: Arial;'>LLMs</h2>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        empty: list[dict[str, str]] = []
        st.session_state.messages = empty

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("prompt?"):
        st.session_state.messages.extend(
            [
                _msg("user", user_input),
                # Add placeholders for AI responses
                *[_msg(f"{model.name} ai", "") for model in models],
            ],
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(handle_all_responses(prompt, user_input))
        finally:
            loop.close()


async def handle_all_responses(
    prompt: ChatPromptTemplate,
    user_input: str,
) -> None:
    tasks = []
    for model in models:
        container = st.empty()
        task = handle_model_responses(prompt, model, user_input, container)
        tasks.append(task)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    response_from_multiple_models()
