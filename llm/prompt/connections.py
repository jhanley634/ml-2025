#! /usr/bin/env python

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


def get_llm_response(prompt: str, model: str = "phi4") -> str:
    ollama_url = "http://localhost:11434"
    llm = ChatOllama(base_url=ollama_url, model=model)
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    return f"{result.content}"


prompt = """
Produce a two-column four-row markdown table using |- + characters,
which maps from CATEGORY to four comma-separated WORDS or phrases within the category.
Be sure to place a ", " COMMA SPACE between each of those four WORDS.
Everything in both columns shall be in ALL CAPS.
"""
examples = [
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    SEEN IN A POTTERY STUDIO
        CLAYGLAZEKILNWHEEL
    THINGS THAT ARE SLIPPERY
        BANANA PEELEELGREASEICE
    NATURAL PRODUCERS OF HEAT
        FIRELIGHTNINGSUNVOLCANO
    CANCEL, AS A PROJECT
        AXECUTDROPSCRAP
""",
        """
| SEEN IN A POTTERY STUDIO          | CLAY, GLAZE, KILN, WHEEL                       |
| THINGS THAT ARE SLIPPERY          | BANANA PEEL, EEL, GREASE, ICE                  |
| NATURAL PRODUCERS OF HEAT         | FIRE, LIGHTNING, SUN, VOLCANO                  |
| CANCEL, AS A PROJECT              | AXE, CUT, DROP, SCRAP                          |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    KINDS OF CARVINGS
        BUSTRELIEFSTATUETORSO
    PILLAR
        BRACEPOSTPROPSUPPORT
    BBQ OFFERING
        DOGLINKRIBWING
    ___NECK
        BOTTLEBREAKGOOSETURTLE
""",
        """
| KINDS OF CARVINGS                 | BUST, RELIEF, STATUE, TORSO                    |
| PILLAR                            | BRACE, POST, PROPS, SUPPORT                    |
| BBQ OFFERING                      | DOG, LINK, RIB, WING                           |
| ___NECK                           | BOTTLE, BREAK, GOOSE, TURTLE                   |
""",
    ),
    (
        """
 Create four groups of four!

4 FoundCategories out of 4

    AWESOME
        DOPEFIRELITSICK
    URL ENDINGS PLUS A LETTER
        COMPMILKNETIORGO
    DEFEAT SOUNDLY
        CREAMLICKPASTESMOKE
    “WILL” CONTRACTIONS WITHOUT THE APOSTROPHE
        HELLILLSHELLWELL
 """,
        """
| AWESOME                           | DOPE, FIRE, LIT, SICK                          |
| URL ENDINGS PLUS A LETTER         | COMP, MILK, NETI, ORGO                         |
| DEFEAT SOUNDLY                    | CREAM, LICK, PASTE, SMOKE                      |
| WILL CONTRACTION, SANS APOSTROPHE | HELL, ILL, SHELL, WELL                         |
""",
    ),
]


def main() -> None:
    for squished, result in examples:
        assert result
        print(get_llm_response(f"{prompt}\n\n{squished}"))


if __name__ == "__main__":
    main()
