#! /usr/bin/env streamlit run --server.runOnSave true --server.headless true --browser.gatherUsageStats false

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from llm.prompt.config import CHAT_PROMPT_TEMPLATE


def single_model() -> None:
    st.markdown(
        "<h2 style='text-align: center; font-family: Arial;'>LLM</h2>",
        unsafe_allow_html=True,
    )

    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

    model = OllamaLLM(model="phi4")
    chain = prompt | model

    if "messages" not in st.session_state:
        empty: list[dict[str, str]] = []
        st.session_state.messages = empty

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("prompt?"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            },
        )
        with st.chat_message("user"):
            st.markdown(user_input)

    with st.chat_message("assistant"):
        response = chain.invoke({"question": user_input})
        st.markdown(response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )


if __name__ == "__main__":
    single_model()
