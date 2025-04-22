#! /usr/bin/env python

prompt = """
Produce a two-column four-row markdown table using |- + characters,
which maps from CATEGORY to four comma-separated WORDS or phrases within the category.
Be sure to place a ", " COMMA SPACE between each of those four WORDS.
Everything in both columns shall be in ALL CAPS.
"""
example1 = """
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
"""
result1 = """
| SEEN IN A POTTERY STUDIO          | CLAY, GLAZE, KILN, WHEEL                       |
| THINGS THAT ARE SLIPPERY          | BANANA PEEL, EEL, GREASE, ICE                  |
| NATURAL PRODUCERS OF HEAT         | FIRE, LIGHTNING, SUN, VOLCANO                  |
| CANCEL, AS A PROJECT              | AXE, CUT, DROP, SCRAP                          |
"""
