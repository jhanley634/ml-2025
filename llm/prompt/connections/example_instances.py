#! /usr/bin/env python


prompt = """
Produce a two-column four-row markdown table using |- + characters,
which maps from CATEGORY to four comma-separated WORDS or phrases within the category.
The second column shall be named WORDS.
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
    WILL CONTRACTION, SANS APOSTROPHE
        HELLILLSHELLWELL
 """,
        """
| AWESOME                           | DOPE, FIRE, LIT, SICK                          |
| URL ENDINGS PLUS A LETTER         | COMP, MILK, NETI, ORGO                         |
| DEFEAT SOUNDLY                    | CREAM, LICK, PASTE, SMOKE                      |
| WILL CONTRACTION, SANS APOSTROPHE | HELL, ILL, SHELL, WELL                         |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    PLAY SOME ELECTRIC GUITAR
        JAMNOODLESHREDSOLO
    QUALITIES OF OVERCOOKED MEAT
        CHEWYDRYSTRINGYTOUGH
    INGREDIENTS IN BUBBLE TEA
        BOBAMILKSUGARTEA
    PLANETS WITH FIRST LETTER CHANGED
        BLUTOCARSDARTHGENUS
""",
        """
| PLAY SOME ELECTRIC GUITAR         | JAM, NOODLE, SHRED, SOLO                       |
| QUALITIES OF OVERCOOKED MEAT      | CHEWY, DRY, STRINGY, TOUGH                     |
| INGREDIENTS IN BUBBLE TEA         | BOBA, MILK, SUGAR, TEA                         |
| PLANETS WITH FIRST LETTER CHANGED | BLUTO, CARS, DARTH, GENUS                      |
""",
    ),
]
