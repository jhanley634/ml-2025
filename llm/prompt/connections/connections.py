#! /usr/bin/env python

import re

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from llm.prompt.connections.example_instances import examples, prompt
from llm.prompt.connections.util import as_df, canonicalize


def get_llm_response(prompt: str, model: str = "phi4") -> str:
    ollama_url = "http://localhost:11434"
    llm = ChatOllama(base_url=ollama_url, model=model)
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    return f"{result.content}"


def main() -> None:
    for squished, result in reversed(examples):
        df = as_df(result)
        md_tbl = df.to_markdown(index=False).replace(":", "-")
        print(canonicalize(re.sub(r" +", " ", md_tbl)))
        response = get_llm_response(f"{prompt}\n\n{squished}")
        print(canonicalize(response))


if __name__ == "__main__":
    main()
