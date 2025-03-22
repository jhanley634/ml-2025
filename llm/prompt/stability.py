#! /usr/bin/env streamlit run --server.runOnSave true --server.headless true --browser.gatherUsageStats false

from dataclasses import dataclass

import streamlit as st
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
]


def _msg(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
    }


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
        st.session_state.messages.append(
            _msg("user", user_input),
        )
        with st.chat_message("user"):
            st.markdown(user_input)

    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
    with st.chat_message("assistant"):
        for model in models:
            chain = prompt | model.model_instance
            response = chain.invoke({"question": user_input})
            st.markdown(f"### {model.name}:")
            st.markdown(response)
            st.session_state.messages.append(
                _msg("assistant", response),
            )


if __name__ == "__main__":
    response_from_multiple_models()
