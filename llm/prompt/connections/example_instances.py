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
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    SILENCE
        CALMHUSHPEACESTILL
    COMPARATIVELY SMALL
        BABYCOMPACTMINUTETOY
    TENNIS COMPETITION UNITS
        GAMEMATCHSETTOURNAMENT
    STARTING WITH SYNONYMS FOR “TEASE”
        KIDNEYMOCKINGBIRDRAZZMATAZZRIBBON
""",
        """
|SILENCE|CALM, HUSH, PEACE, STILL|
|COMPARATIVELY SMALL|BABY, COMPACT, MINUTE, TOY|
|TENNIS COMPETITION UNITS|GAME, MATCH, SET, TOURNAMENT|
|STARTING WITH SYNONYMS FOR “TEASE”|KIDNEY, MOCKINGBIRD, RAZZMATAZZ, RIBBON|
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    TAROT MINOR ARCANA SUITS
        CUPSPENTACLESSWORDSWANDS
    NOT INCLUDING
        BESIDESBUTEXCEPTSAVE
    HOMOPHONES OF GEMSTONES
        CHORALOPELPURLQUARTS
    GET BETTER, AS A BROKEN BONE
        HEALKNITMENDRECOVER
""",
        """
| TAROT MINOR ARCANA SUITS          | CUPS, PENTACLES, SWORDS, WANDS                 |
| NOT INCLUDING                     | BESIDES, BUT, EXCEPT, SAVE                     |
| HOMOPHONES OF GEMSTONES           | CHORAL, OPEL, PURL, QUARTS                     |
| GET BETTER, AS A BROKEN BONE      | HEAL, KNIT, MEND, RECOVER                      |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    COMMIT TO PAPER
        AUTHORCOMPOSEPENWRITE
    NEEDS FOR PLAYING YAHTZEE
        CUPDICEPENCILSCORECARD
    HEROES OF ACTION MOVIE FRANCHISES
        BONDJONESOCEANWICK
    ___STICK
        CANDLECHOPJOYYARD
""",
        """
| COMMIT TO PAPER                   | AUTHOR, COMPOSE, PEN, WRITE                    |
| NEEDS FOR PLAYING YAHTZEE         | CUP, DICE, PENCIL, SCORECARD                   |
| HEROES OF ACTION MOVIE FRANCHISES | BOND, JONES, OCEAN, WICK                       |
| ___STICK                          | CANDLE, CHOP, JOY, YARD                        |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    TV DISPLAY SETTINGS
        BRIGHTNESSCOLORCONTRASTTINT
    RESULTS OF SOME ARITHMETIC
        DIFFERENCEPRODUCTQUOTIENTSUM
    FUZZY, AS A MEMORY
        DIMFAINTREMOTEVAGUE
    WINDOW TREATMENTS IN THE SINGULAR
        BLINDDRAPESHADESHUTTER
""",
        """
| TV DISPLAY SETTINGS               | BRIGHTNESS, COLOR, CONTRAST, TINT              |
| RESULTS OF SOME ARITHMETIC        | DIFFERENCE, PRODUCT, QUOTIENT, SUM             |
| FUZZY, AS A MEMORY                | DIM, FAINT, REMOTE, VAGUE                      |
| WINDOW TREATMENTS IN THE SINGULAR | BLIND, DRAPES, SHADE, SHUTTER                  |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    LETTER SIGN-OFFS
        BESTCHEERSLOVESINCERELY
    WITHOUT
        ABSENTMINUSSANSWANTING
    VIGOR
        BEANSENERGYPEPZIP
    ___ STRIP
        BACONCOMICLANDINGSUNSET
""",
        """
| LETTER SIGN-OFFS                  | BEST, CHEERS, LOVE, SINCERELY                  |
| WITHOUT                           | ABSENT, MINUS, SANS, WANTING                   |
| VIGOR                             | BEANS, ENERGY, PEP, ZIP                        |
| ___ STRIP                         | BACON, COMIC, LANDINGS, SUNSET                 |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    MAKE HAPPY
        DELIGHTPLEASESUITTICKLE
    EVADE
        DODGEDUCKSHAKESKIRT
    COMMON VIDEO GAME FEATURES
        BOSSHEALTHLEVELPOWER-UP
    MOTHER ___
        EARTHGOOSEMAY ISUPERIOR
""",
        """
| MAKE HAPPY                        | DELIGHT, PLEASE, SUIT, TICKLE                  |
| EVADE                             | DODGE, DUCK, SHAKE SKIRT,                      |
| COMMON VIDEO GAME FEATURES        | BOSS, HEALTH, LEVEL, POWER-UP                  |
| MOTHER ___                        | EARTH, GOOSE, MAY I, SUPERIOR                  |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    THINGS TRACKED BY WEB ANALYTICS
        CLICKHITPAGE VIEWVISIT
    THINGS YOU CAN DO WITH YOUR LIPS
        CURLPUCKERPURSESMACK
    PLACES TO FIND PAPER MONEY
        ATMCASH REGISTERTIP JARWALLET
    REBOUND
        BANKBOUNCECAROMRICOCHET
""",
        """
| THINGS TRACKED BY WEB ANALYTICS   | CLICK, HIT, PAGE VIEW, VISIT                   |
| THINGS YOU CAN DO WITH YOUR LIPS  | CURL, PUCKER, PURSE, SMACK                     |
| PLACES TO FIND PAPER MONEY        | ATM, CASH REGISTER, TIP JAR, WALLET            |
| REBOUND                           | BANK, BOUNCE, CAROM, RICOCHET                  |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    ENDING WITH COLORS
        EVERGREENINFRAREDMARIGOLDQUICKSILVER
    PLACES TO SHOP
        BAZAARFAIRMARKETOUTLET
    KINDS OF PIZZA
        HAWAIIANPLAINSUPREMEVEGGIE
    ___ CLEANER
        BATHROOMDRYPIPEVACUUM

""",
        """
| ENDING WITH COLORS                | EVERGREEN, INFRARED, MARIGOLD, QUICKSILVER     |
| PLACES TO SHOP                    | BAZAAR, FAIR, MARKET, OUTLET                   |
| KINDS OF PIZZA                    | HAWAIIAN, PLAIN, SUPREME, VEGGIE               |
| ___ CLEANER                       | BATHROOM, DRY, PIPE, VACUUM                    |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    ABSORB USING CAPILLARY ACTION
        DRAWPULLSUCKWICK
    STARTING WITH SILENT LETTERS
        GNOMEKNEEMNEMONICPSYCHE
    GREEK PREFIXES
        HYPERKILOMETANEO
    TITULAR ANIMALS OF FILM
        BABEBOLTDUMBOTED
""",
        """
| ABSORB USING CAPILLARY ACTION     | DRAW, PULL, SUCK, WICK                         |
| STARTING WITH SILENT LETTERS      | GNOME, KNEE, MNEMONIC, PSYCHE                  |
| GREEK PREFIXES                    | HYPER, KILO, META, NEO                         |
| TITULAR ANIMALS OF FILM           | BABE, BOLT, DUMBO, TED                         |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    TYPES OF SNOW
        CRUSTICEPOWDERSLUSH
    RATIONALE
        BASISFOUNDATIONGROUNDSREASON
    LAST WORDS OF FAMOUS OPERA TITLES
        BESSBUTTERFLYFLUTESEVILLE
    REAL ___
        DEALESTATEMADRIDWORLD
""",
        """
| TYPES OF SNOW                     | CRUST, ICE, POWDER, SLUSH                      |
| RATIONALE                         | BASIS, FOUNDATION, GROUNDS, REASON             |
| LAST WORDS OF FAMOUS OPERA TITLES | BESS, BUTTERFLY, FLUTE, SEVILLE                |
| REAL ___                          | DEAL, ESTATE, MADRID, WORLD                    |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    KINDS OF PLANTS
        HERBSHRUBTREEVINE
    DISCONTINUE
        DISSOLVEENDSCRAPSUNSET
    ENDING WITH BUILDING MATERIALS
        HOLLYWOODHOURGLASSKUBRICKNEUROPLASTIC
    ASSOCIATED WITH BULLS
        MICHAEL JORDANRODEOTAURUSWALL STREET
""",
        """
| KINDS OF PLANTS                   | HERB, SHRUB, TREE, VINE                        |
| DISCONTINUE                       | DISSOLVE, END, SCRAP, SUNSET                   |
| ENDING WITH BUILDING MATERIALS    | HOLLYWOOD, HOURGLASS, KUBRICK, NEUROPLASTIC    |
| ASSOCIATED WITH BULLS             | MICHAEL JORDAN, RODEO, TAURUS, WALL STREET     |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    MEMBER OF A KINGDOM IN TAXONOMY
        ANIMALBACTERIAFUNGUSPLANT
    GRADUATED INSTRUMENTS
        BEAKERPROTRACTORRULERSYRINGE
    KINDS OF PENGUINS
        CHINSTRAPEMPERORKINGMACARONI
    “E” THINGS
        COMMERCEMAILSCOOTERSIGNATURE""",
        """
| MEMBER OF A KINGDOM IN TAXONOMY   | ANIMAL, BACTERIA, FUNGUS, PLANT                |
| GRADUATED INSTRUMENTS             | BEAKER, PROTRACTOR, RULER, SYRINGE             |
| KINDS OF PENGUINS                 | CHINSTRAP, EMPEROR, KINGDOM, MACARONI          |
| “E” THINGS                        | COMMERCE, EMAIL, SCOOTER, SIGNATURE            |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    TASKS FOR A STUDENT
        ASSIGNMENTDRILLEXERCISELESSON
    ENCOURAGING RESPONSES IN A GUESSING GAME
        ALMOSTCLOSENOT QUITEWARM
    UP FOR ANYTHING
        EASYFLEXIBLEGAMEOPEN
    WHAT “A” MIGHT MEAN
        AREAATHLETICEXCELLENTONE
""",
        """
| TASKS FOR A STUDENT               | ASSIGNMENT, DRILL, EXERCISE, LESSON            |
| GUESS GAME ENCOURAGING RESPONSES  | ALMOST, CLOSE, NOT QUITE, WARM                 |
| UP FOR ANYTHING                   | EASY, FLEXIBLE, GAME, OPEN                     |
| WHAT “A” MIGHT MEAN               | AREA, ATHLETIC, EXCELLENT, ONE                 |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    FUNDRAISING EVENT
        BALLBENEFITFUNCTIONGALA
    HOMOPHONES OF PARTS OF THE FOOT
        BAWLHEALSOULTOW
    FEATURES OF A TOOTHED WHALE
        BLUBBERFLIPPERFLUKEMELON
    SPACES ON A MONOPOLY BOARD
        AVENUECHANCERAILROADUTILITY
""",
        """
| FUNDRAISING EVENT                 | BALL, BENEFIT, FUNCTION, GALA                  |
| HOMOPHONES OF PARTS OF THE FOOT   | BAWL, HEAL, SOUL, TOW                          |
| FEATURES OF A TOOTHED WHALE       | BLUBBER, FLIPPER, FLUKE, MELON                 |
| SPACES ON A MONOPOLY BOARD        | AVENUE, CHANCE, RAILROAD, UTILITY              |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    SEEN IN A BARN
        BALEHORSEPITCHFORKTROUGH
    WORDS BEFORE “BED”
        CANOPYDAYMURPHYWATER
    ACCOUNT BOOK
        LEDGERLOGRECORDREGISTER
    DETECTIVES OF KID-LIT
        BROWNDREWHARDYHOLMES
""",
        """
| SEEN IN A BARN                    | BALE, HORSE, PITCHFORK, TROUGH                 |
| WORDS BEFORE “BED”                | CANOPY, DAY, MURPHY, WATER                     |
| ACCOUNT BOOK                      | LEDGER, LOG, RECORD, REGISTER                  |
| DETECTIVES OF KID-LIT             | BROWN, DREWE, HARDY, HOLMES                    |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    PROHIBIT, AS ENTRY
        BARBLOCKDENYREFUSE
    FOLDERS ON A MAC
        DESKTOPMUSICPICTURESTRASH
    MEDICINE FORMATS
        CREAMPATCHSPRAYTABLET
    THINGS THAT OPEN LIKE A CLAM
        CLAMCOMPACTLAPTOPWAFFLE IRON
""",
        """
| PROHIBIT, AS ENTRY                | BAR, BLOCK, DENY, REFUSE                       |
| FOLDERS ON A MAC                  | DESKTOP, MUSIC, PICTURES, TRASH                |
| MEDICINE FORMATS                  | CREAM, PATCH, SPRAY, TABLET                    |
| THINGS THAT OPEN LIKE A CLAM      | CLAMP, COMPACT, LAPTOP, WAFFLE IRON            |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    CHARACTERS WITH GREEN SKIN
        ELPHABAGRINCHHULKSHREK
    FAMOUS RIDDLE-GIVERS
        BRIDGE TROLLMAD HATTERRIDDLERSPHINX
    FEATURES OF THE NATIONAL MALL IN D.C.
        CAPITOLMALLOBELISKPOOL
    FINE PRINT
        ASTERISKCATCHCONDITIONSTRINGS
""",
        """
| CHARACTERS WITH GREEN SKIN        | ELPHABA, GRINCH, HULK, SHREK                   |
| FAMOUS RIDDLE-GIVERS              | BRIDGE TROLL, MAD HATTER, RIDDLER, SPHINX      |
| FEATURES OF D.C.'S NATIONAL MALL  | CAPITOL, MALL, OBELISK, POOL                   |
| FINE PRINT                        | ASTERISK, CATCH, CONDITIONAL, STRINGS          |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    QUITE THE PARTY
        BASHBLASTBLOWOUTRAVE
    ONE’S CONSTITUTION
        CHARACTERFIBERMAKEUPNATURE
    BRITISH IMPERIAL UNITS OF WEIGHT
        DRAMOUNCEPOUNDSTONE
    WHAT “CAT’S EYE” CAN BE USED TO DESCRIBE
        EYELINERGLASSESMARBLENEBULA
""",
        """
| QUITE THE PARTY                   | BASH, BLAST, BLOWOUT, RAVE                     |
| ONE’S CONSTITUTION                | CHARACTER, FIBER, MAKEUP, NATURE               |
| BRITISH IMPERIAL UNITS OF WEIGHT  | DRAM, OUNCE, POUND, STONE                      |
| WHAT “CAT’S EYE” MIGHT DESCRIBE   | EYELINER, GLASSES, MARBLE, NEBULA              |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    NEW YORK SPORTS TEAM MEMBERS
        JETMETNETRANGER
    BABY GEAR
        BIBBOTTLEMONITORSTROLLER
    KINDS OF PANTS MINUS "S"
        CAPRIJEANJOGGERSLACK
    BLACK WOMEN AUTHORS
        BUTLERGAYHOOKSWALKER
""",
        """
| NEW YORK SPORTS TEAM MEMBERS      | JET, MET, NET, RANGER                          |
| BABY GEAR                         | BIB, BOTTLE, MONITOR, STROLLER                 |
| KINDS OF PANTS MINUS "S"          | CAPRI, JEAN, JOGGER, SLACK                     |
| BLACK WOMEN AUTHORS               | BUTLER, GAY, HOOKS, WALKER                     |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    DOPPELGÄNGER
        CLONEDOUBLERINGERTWIN
    PLAYING CARDS
        ACEJACKKINGQUEEN
    ___ MAIL
        CHAINELECTRONICJUNKSNAIL
    EAR PIERCING SITES
        CONCHHELIXLOBEROOK
""",
        """
| DOPPELGÄNGER                      | CLONE, DOUBLE, RINGER, TWIN                    |
| PLAYING CARDS                     | ACE, JACK, KING, QUEEN                         |
| ___ MAIL                          | CHAIN, ELECTRONIC, JUNK, SNAIL                 |
| EAR PIERCING SITES                | CONCH, HELIX, LOBE, ROOK                       |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    ITEMS IN A LINEN CLOSET
        PILLOWCASESHEETTOWELWASHCLOTH
    LINGERIE
        GARTERHOSESLIPTEDDY
    DIAMETRIC
        COUNTEROPPOSITEPOLARREVERSE
    CARD GAMES WITH FIRST LETTER CHANGED
        DINFRIDGEGUMMYJOKER
""",
        """
| ITEMS IN A LINEN CLOSET           | PILLOWCASE, SHEET, TOWEL, WASHCLOTH            |
| LINGERIE                          | GARTER, HOSE, SLIP, TEDDY                      |
| DIAMETRIC                         | COUNTER, OPPOSITE, POLAR, REVERSE              |
| CARD GAME WITH 1ST LETTER CHANGED | DIN, FRIDGE, GUMMY, JOKER                      |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    GUITAR PLAYING TECHNIQUES
        BENDPICKSLIDESTRUM
    WHAT CHARACTERS WERE TRANSFORMED INTO IN “BEAUTY AND THE BEAST”
        BEASTCANDELABRACLOCKTEACUP
    ROUND FLAT THINGS
        COASTERFRISBEEPANCAKERECORD
    AIRPORT FEATURES
        CAROUSELFOOD COURTGATELOUNGE
""",
        """
| GUITAR PLAYING TECHNIQUES         | BEND, PICK, SLIDE, STRUM                       |
| WHAT CHARACTERS BECAME IN BEAUTY  | BEAST, CANDELABRA, CLOCK, TEACUP               |
| ROUND FLAT THINGS                 | COASTER, FRISBEE, PANCAKE, RECORD              |
| AIRPORT FEATURES                  | CAROUSEL, FOOD COURT, GATE, LOUNGE             |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    CONNECT
        BRIDGEJOINLINKUNITE
    PARTS OF A BIKE
        CHAINPEDALSADDLEWHEEL
    BEST PICTURE WINNERS SINCE 2000
        CHICAGOCRASHGLADIATORMOONLIGHT
    MUSIC GENRES PLUS A LETTER
        BLUESTPOPEROCKYSKAT
""",
        """
| CONNECT                           | BRIDGE, JOIN, LINK, UNITE                      |
| PARTS OF A BIKE                   | CHAIN, PEDALS, SADDLE, WHEEL                   |
| BEST PICTURE WINNERS SINCE 2000   | CHICAGO, CRASH, GLADIATOR, MOONLIGHT           |
| MUSIC GENRES PLUS A LETTER        | BLUES, POP, ROCKY, SKAT                        |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    RODS
        BATCLUBSTAFFSTICK
    THEY'RE ON A ROLL!
        FOILRIBBONTAPETOILET PAPER
    WORDS BEFORE “SHACK”
        CADDYLOVERADIOSHAKE
    COMMON SWAG ITEMS
        HATTEETOTEWATER BOTTLE
""",
        """
| RODS                              | BAT, CLUB, STAFF, STICK                        |
| THEY'RE ON A ROLL!                | FOIL, RIBBON, TAPE, TOILET PAPER               |
| WORDS BEFORE “SHACK”              | CADDY, LOVE, RADIO, SHAKE                      |
| COMMON SWAG ITEMS                 | HAT, TEE, TOTE, WATER BOTTLE                   |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    ALCOVE
        CAVITYHOLLOWNOOKRECESS
    VERBS IN BREADMAKING
        FERMENTPROOFRESTRISE
    WAYS TO RECOGNIZE ACHIEVEMENT
        CERTIFICATEMEDALPLAQUETROPHY
    THINGS YOU CAN BLOW
        BUBBLEFUSEKISSRASPBERRY
""",
        """
| ALCOVE                            | CAVITY, HOLLOW, NOOK, RECESS                   |
| VERBS IN BREADMAKING              | FERMENT, PROOF, REST, RISE                     |
| WAYS TO RECOGNIZE ACHIEVEMENT     | CERTIFICATE, MEDAL, PLAQUE, TROPHY             |
| THINGS YOU CAN BLOW               | BUBBLE, FUSE, KISS, RASPBERRY                  |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    ___ FUND
        HEDGEMUTUALSLUSHTRUST
    LOCAL WATERING HOLE
        DIVEESTABLISHMENTHAUNTJOINT
    COMPETE IN A MODERN PENTATHLON
        FENCERIDESHOOTSWIM
    ENSURE, AS A VICTORY
        CINCHGUARANTEEICELOCK
""",
        """
| ___ FUND                          | HEDGE, MUTUAL, SLUSH, TRUST                    |
| LOCAL WATERING HOLE               | DIVE, ESTABLISHMENT, HAUNT, JOINT              |
| COMPETE IN A MODERN PENTATHON     | FENCE, RIDE, SHOOT, SWIM                       |
| ENSURE, AS A VICTORY              | CINCH, GUARANTEE, ICE, LOCK                    |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    EXPEDITION
        JOURNEYODYSSEYQUESTVOYAGE
    HOLD DEAR
        ESTEEMPRIZETREASUREVALUE
    WORDS WHOSE ONLY VOWEL IS “Y”
        MYRRHNYMPHRHYTHMSPHYNX
    NAMES ENDING IN “K” PLUS WORD
        FRANKINCENSEJACKPOTMARKDOWNNICKNAME
""",
        """
| EXPEDITION                        | JOURNEY, ODYSSEY, QUEST, VOYAGE                |
| HOLD DEAR                         | ESTEEM, PRIZE, TREASURE, VALUE                 |
| WORDS WHOSE ONLY VOWEL IS “Y”     | MYRRH, NYMPH, RHYTHM, SPHINX                   |
| NAMES ENDING IN “K” PLUS WORD     | FRANKINCENSE, JACKPOT, MARKDOWN, NICKNAME      |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    BE OSTENTATIOUS
        GRANDSTANDPOSTURESHOWBOATSWAGGER
    COPACETIC
        FINEHUNKY-DORYOKSWELL
    KINDS OF BRACELETS
        CHARMFRIENDSHIPIDTENNIS
    THINGS YOU CAN PRACTICE
        LAWMEDICINESELF-CAREWITCHCRAFT
""",
        """
| BE OSTENTATIOUS                   | GRANDSTAND, POSTURE, SHOWBOAT, SWAGGER         |
| COPACETIC                         | FINE, HUNKY-DORY, OK, SWELL                    |
| KINDS OF BRACELETS                | CHARM, FRIENDSHIP, ID, TENNIS                  |
| THINGS YOU CAN PRACTICE           | LAW, MEDICINE, SELF-CARE, WITCHCRAFT           |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    PARTS OF A SONG
        BRIDGECHORUSHOOKREFRAIN
    IMAGERY IN MAGRITTE PAINTINGS
        APPLEBOWLERCLOUDPIPE
    THINGS IN AN ENTRYWAY
        BENCHCOAT RACKCONSOLERUNNER
    SUPPORT AUDIBLY
        CHEERCLAPROOTWHISTLE
""",
        """
| PARTS OF A SONG                   | BRIDGE, CHORUS, HOOK, REFRAIN                  |
| IMAGERY IN MAGRITTE PAINTINGS     | APPLE, BOWLER, CLOUD, PIPE                     |
| THINGS IN AN ENTRYWAY             | BENCH, COAT RACK, CONSOL, RUNNER               |
| SUPPORT AUDIBLY                   | CHEER, CLAP, ROAR, WHISTLE                     |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    WAYS TO MODIFY A CAR'S EXTERIOR
        BUMPERGRILLERIMSPOILER
    PLUMBING EQUIPMENT
        PIPEPLUNGERSNAKEWRENCH
    PRECIPICE
        BRINKCUSPEVEVERGE
    BEST FEMALE ROCK PERFORMANCE GRAMMY WINNERS
        APPLECROWSUMMERTURNER
""",
        """
| WAYS TO MODIFY A CAR'S EXTERIOR   | BUMPER, GRILL, RIM, SPOILER                    |
| PLUMBING EQUIPMENT                | PIPE, PLUNGER, SNAKE, WRENCH                   |
| PRECIPICE                         | BRINK, CUSP, EVE, VERGE                        |
| BEST FEMALE ROCKER GRAMMY WINNERS | APPLE, CROW, SUMMER, TURNER                    |
""",
    ),
    (
        """
Create four groups of four!

4 FoundCategories out of 4

    NEWSPAPER JOBS
        COLUMNISTEDITORPHOTOGRAPHERREPORTER
    EVERYDAY
        COMMONREGULARROUTINESTANDARD
    ENDING WITH KINDS OF DOGS
        NEWSHOUNDSHADOWBOXERSNICKERDOODLETRENDSETTER
    WHAT “CON” MIGHT MEAN
        CONVENTIONCRIMINALDRAWBACKSWINDLE
""",
        """
| NEWSPAPER JOBS               | COLUMNIST, EDITOR, PHOTOGRAPHER, REPORTER          |
| EVERYDAY                     | COMMON, REGULAR, ROUTINE, STANDARD                 |
| ENDING WITH KINDS OF DOGS    | NEWSHOUND, SHADOWBOXER, SNICKERDOODLE, TRENDSETTER |
| WHAT “CON” MIGHT MEAN        | CONVENTION, CRIMINAL, DRAWBACK, SWINDLE            |
""",
    ),
    (
        """
""",
        """
""",
    ),
]
