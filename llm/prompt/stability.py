#! /usr/bin/env streamlit run --server.runOnSave true --server.headless true --browser.gatherUsageStats false

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import streamlit as st
from beartype import beartype
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from llm.prompt.config import CHAT_PROMPT_TEMPLATE


@dataclass
class LLM:
    name: str
    model_instance: OllamaLLM


def _get_model(name: str) -> LLM:
    return LLM(name, OllamaLLM(model=name))


models = [
    _get_model("gemma"),
    _get_model("phi4"),
    # _get_model("gemma3:12b"),
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
) -> AsyncGenerator[str, None]:
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
    container: st._DeltaGenerator,
) -> None:
    async for token in get_streaming_response_from_model(prompt, model, user_input):
        msg = {"role": f"{model.name} ai", "content": token}
        st.session_state.messages.append(msg)
        container.write(f"**{model.name}**:\n\n{token}")


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

        container = st.empty()

        async def handle_all_responses() -> None:
            tasks = [
                handle_model_responses(prompt, model, user_input, container) for model in models
            ]
            await asyncio.gather(*tasks)

        asyncio.run(handle_all_responses())


if __name__ == "__main__":
    response_from_multiple_models()
