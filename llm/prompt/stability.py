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
    if f"{model.name}_messages" not in st.session_state:
        st.session_state[f"{model.name}_messages"] = []

    async for token in get_streaming_response_from_model(prompt, model, user_input):
        msg = _msg(model.name, token)

        st.session_state[f"{model.name}_messages"].append(msg)

        text = "".join(elt["content"] for elt in st.session_state[f"{model.name}_messages"])
        container.markdown(text)


async def handle_all_responses(
    prompt: ChatPromptTemplate,
    user_input: str,
) -> None:
    containers = {model.name: st.empty() for model in models}
    tasks = [
        handle_model_responses(prompt, model, user_input, containers[model.name])
        for model in models
    ]
    await asyncio.gather(*tasks)


def response_from_multiple_models() -> None:
    st.markdown(
        "<h2 style='text-align: center; font-family: Arial;'>LLMs</h2>",
        unsafe_allow_html=True,
    )

    user_input = st.chat_input("prompt?")

    if user_input:
        for model in models:
            if f"{model.name}_messages" not in st.session_state:
                st.session_state[f"{model.name}_messages"] = []

        prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
        prompt.append(f"{user_input} \n\n")

        asyncio.run(handle_all_responses(prompt, user_input))


if __name__ == "__main__":
    response_from_multiple_models()
