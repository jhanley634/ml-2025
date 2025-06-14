#! /usr/bin/env python

import re
from difflib import unified_diff

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from llm.prompt.connections.example_instances import examples, prompt
from llm.prompt.connections.util import as_df, canonicalize, validate


def get_llm_response(prompt: str, model: str = "phi4") -> str:
    ollama_url = "http://localhost:11434"
    llm = ChatOllama(base_url=ollama_url, model=model)
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    return f"{result.content}"


def main() -> None:
    for squished, result in reversed(examples):
        if not squished.strip():
            continue
        df = as_df(result)
        md_tbl = canonicalize(df.to_markdown(index=False, tablefmt="github"))
        response = get_llm_response(f"{prompt}\n\n{squished}")
        response = "\n".join(filter(lambda line: "|" in line, response.split("\n")))
        response = canonicalize(response)
        print(re.sub(r"^|$", "|", response, flags=re.MULTILINE))

        delta = unified_diff(
            md_tbl.split("\n"),
            response.split("\n"),
        )
        print("\n".join(delta), "\n\n")

        validate(df)


if __name__ == "__main__":
    main()
