#!/usr/bin/env python3
"""Hand-assembled labels for chunk_007, one label per post index [0..199].

Each index maps to exactly one of NOISE_MEME / GENUINE_OPINION /
FINANCIAL_CLAIM / ANALYSIS_DD. Index -> label kept explicit to guarantee
alignment with the post order and to avoid off-by-one drift.
"""

LABELS_BY_INDEX = {
    0: "NOISE_MEME",        # chart caption only, no argument
    1: "NOISE_MEME",        # meme "shorts rough 2020"
    2: "NOISE_MEME",        # "wu tang formation" joke
    3: "NOISE_MEME",        # meme
    4: "NOISE_MEME",        # "The Pope is infected" off-topic
    5: "NOISE_MEME",        # satire lifestyle
    6: "NOISE_MEME",        # meme nostalgia
    7: "FINANCIAL_CLAIM",   # MSFT earnings date 27th, holds 11/06 call, asking
    8: "ANALYSIS_DD",       # consensus price targets + yield assumption
    9: "FINANCIAL_CLAIM",   # "22,000 puts at $30 for RKT" concrete number
    10: "NOISE_MEME",       # gain brag
    11: "ANALYSIS_DD",      # GME deep-value + squeeze thesis w/ numbers
    12: "NOISE_MEME",       # meme trailer
    13: "FINANCIAL_CLAIM",  # "-$52k since Oct 14th" own position number
    14: "GENUINE_OPINION",  # ONEM bullish call + position, no reasoning
    15: "GENUINE_OPINION",  # "buying JCP calls for earnings" intent
    16: "GENUINE_OPINION",  # spy verticals "free money" opinion/stance
    17: "NOISE_MEME",       # "BULLS R FUK ... CONFIRMED" meme
    18: "NOISE_MEME",       # "PRINTERS WILL GO BRRRRRR" hype
    19: "NOISE_MEME",       # Jerome Powell copypasta storytime
    20: "NOISE_MEME",       # meme
    21: "ANALYSIS_DD",      # DKNG valuation w/ revenue & margin math
    22: "NOISE_MEME",       # meme
    23: "GENUINE_OPINION",  # "THIS IS THE BOTTOM ... SPY 1000c" bullish
    24: "NOISE_MEME",       # Ronaldo/Messi spam
    25: "NOISE_MEME",       # BA bailout joke
    26: "NOISE_MEME",       # broker question joke
    27: "NOISE_MEME",       # asking for DD, no info/opinion of own
    28: "NOISE_MEME",       # meta complaint
    29: "FINANCIAL_CLAIM",  # Apple China store shutdown (insider fact)
    30: "NOISE_MEME",       # shitpost
    31: "NOISE_MEME",       # shitpost
    32: "FINANCIAL_CLAIM",  # Robinhood day-trader warning (news fact)
    33: "NOISE_MEME",       # meme
    34: "FINANCIAL_CLAIM",  # "LEN 80c 9/18" position specifics
    35: "NOISE_MEME",       # "20 years old addicted to gambling"
    36: "NOISE_MEME",       # loss joke
    37: "FINANCIAL_CLAIM",  # Ark price target 15,000 (relayed fact)
    38: "ANALYSIS_DD",      # Hertz thesis: company warned stockholders lose all
    39: "GENUINE_OPINION",  # FOMO, considering calls, vaccine hope
    40: "NOISE_MEME",       # "Tesla changed my life" storytime
    41: "GENUINE_OPINION",  # holds TACO calls to the end
    42: "GENUINE_OPINION",  # "buy a shit ton of NOK"
    43: "NOISE_MEME",       # shitpost
    44: "NOISE_MEME",       # "parents put savings in Nokia" joke
    45: "NOISE_MEME",       # meta meme
    46: "NOISE_MEME",       # process question on PDT/spreads
    47: "FINANCIAL_CLAIM",  # CVNA earnings 10/29, up 200% (concrete facts)
    48: "GENUINE_OPINION",  # "LONG GOLD TMRW" no DD given
    49: "ANALYSIS_DD",      # strangle/IV-crush strategy w/ risk mgmt
    50: "NOISE_MEME",       # app question
    51: "ANALYSIS_DD",      # market selloff analysis w/ data
    52: "NOISE_MEME",       # "puts are safe" reaction
    53: "NOISE_MEME",       # "what to yolo next"
    54: "ANALYSIS_DD",      # AIMT bull case: drug approval, short interest
    55: "NOISE_MEME",       # process question on PMCC collateral
    56: "FINANCIAL_CLAIM",  # iron butterfly HYG position numbers
    57: "NOISE_MEME",       # gain brag
    58: "FINANCIAL_CLAIM",  # AIMT PDUFA late January (fact)
    59: "NOISE_MEME",       # political satire
    60: "ANALYSIS_DD",      # TSLA vs $20 oil business-model bear case
    61: "NOISE_MEME",       # meme
    62: "NOISE_MEME",       # battlestation copypasta
    63: "NOISE_MEME",       # "big boy yet"
    64: "ANALYSIS_DD",      # SPCE bear thesis, 5 numbered points, sources
    65: "NOISE_MEME",       # shitpost
    66: "FINANCIAL_CLAIM",  # SPY 337p, -30% position
    67: "NOISE_MEME",       # Robinhood hackers meme
    68: "GENUINE_OPINION",  # "if you haven't lost 10k you're up" ethos
    69: "GENUINE_OPINION",  # "MSFT pump time" hype call
    70: "FINANCIAL_CLAIM",  # "90k of puts on SPY" position number
    71: "NOISE_MEME",       # gain joke
    72: "ANALYSIS_DD",      # locusts/tea supply chain w/ sources
    73: "GENUINE_OPINION",  # "bet majority of portfolio on AAPL calls"
    74: "NOISE_MEME",       # "got a bunch of Disney calls" gain
    75: "FINANCIAL_CLAIM",  # MVIS $4C at .18, 90 contracts (position fact)
    76: "ANALYSIS_DD",      # Battery day info: next-gen batteries, buy TSLA
    77: "NOISE_MEME",       # cryptic bullish note
    78: "NOISE_MEME",       # "too broke to bet"
    79: "NOISE_MEME",       # grizzly bear meme
    80: "NOISE_MEME",       # "new subscriber" meme
    81: "NOISE_MEME",       # daily thread
    82: "NOISE_MEME",       # "Us vs the Mods" meta + position dump
    83: "FINANCIAL_CLAIM",  # "$100 HTZ play sold for $6k" (trade fact)
    84: "NOISE_MEME",       # asking best yolos
    85: "FINANCIAL_CLAIM",  # Barron's Dow 30k article w/ index numbers
    86: "NOISE_MEME",       # "that's all it's worth"
    87: "FINANCIAL_CLAIM",  # Tesla sales staff want higher pay (fact)
    88: "ANALYSIS_DD",      # covid r0/mortality data -> VIX call (reasoned)
    89: "NOISE_MEME",       # "winner or looser" yolo
    90: "GENUINE_OPINION",  # sentiment/ideology, "time for SPY to bleed"
    91: "GENUINE_OPINION",  # "TSLA below 500 today" directional call
    92: "NOISE_MEME",       # "people getting into PLTR"
    93: "NOISE_MEME",       # "most diamond hands"
    94: "NOISE_MEME",       # "why you buy options on foreign ETFs" reaction
    95: "NOISE_MEME",       # "burn red candles to celebrate"
    96: "NOISE_MEME",       # copypasta raid CNBC
    97: "FINANCIAL_CLAIM",  # China quarantine/market data, Hang Seng -6%
    98: "FINANCIAL_CLAIM",  # court stops tree felling Tesla Gigafactory
    99: "NOISE_MEME",       # shitpost
    100: "NOISE_MEME",      # "what sobriety looks like"
    101: "GENUINE_OPINION",  # "Nasdaq only fell 10%" bearish jab
    102: "NOISE_MEME",      # "Captain WSB onboard"
    103: "FINANCIAL_CLAIM", # "MSFT Jan23 2022 175 synthetic CC" position
    104: "NOISE_MEME",      # portfolio photo, no info
    105: "ANALYSIS_DD",     # GME short-break-even model w/ algebra
    106: "NOISE_MEME",      # "learned something smart" generic loss
    107: "NOISE_MEME",      # "buy high sell low" generic
    108: "GENUINE_OPINION", # "SDS puts ... cash in on bear gang" intent
    109: "NOISE_MEME",      # Elon profile pic NWO
    110: "ANALYSIS_DD",     # NHL Vegas catalyst for MGM
    111: "NOISE_MEME",      # "RH foreshadowing"
    112: "ANALYSIS_DD",     # market-tank prediction w/ Fed + red futures reasons
    113: "NOISE_MEME",      # "companies that benefit from corona" vague
    114: "NOISE_MEME",      # yolo brag
    115: "NOISE_MEME",      # PLTR FOMO taunt
    116: "FINANCIAL_CLAIM", # mention-count data table (concrete stats)
    117: "FINANCIAL_CLAIM", # INTC call position, avg down numbers
    118: "NOISE_MEME",      # Boeing Space Force hype
    119: "FINANCIAL_CLAIM", # TSLA S&P500 inclusion (relayed fact)
    120: "NOISE_MEME",      # "2 Model S on margin" flex
    121: "FINANCIAL_CLAIM", # S&P/silver ratio chart, SLV call position
    122: "NOISE_MEME",      # "officially cool"
    123: "FINANCIAL_CLAIM", # Disney+ 25M subs announcement (fact)
    124: "NOISE_MEME",      # earnings thread
    125: "NOISE_MEME",      # "buy yelp reviews" spam
    126: "NOISE_MEME",      # "ready for bloody Monday" reaction
    127: "FINANCIAL_CLAIM", # Sony statement denies rumor (fact)
    128: "NOISE_MEME",      # "hold or cut losses"
    129: "NOISE_MEME",      # meme about TSLA puts
    130: "GENUINE_OPINION", # "betting on a Ford turnaround"
    131: "FINANCIAL_CLAIM", # China stops buying soybeans (headline fact)
    132: "FINANCIAL_CLAIM", # "$15k in SPY 275C for 4/1" position
    133: "NOISE_MEME",      # "tell me how retarded I am"
    134: "NOISE_MEME",      # "Captain WSB onboard"
    135: "NOISE_MEME",      # HSA/options tax process question
    136: "NOISE_MEME",      # "these are your puts"
    137: "GENUINE_OPINION", # "sooner or later it's gonna crash"
    138: "ANALYSIS_DD",     # SPY vs CCL puts risk comparison
    139: "FINANCIAL_CLAIM", # "Robinhood confirms circuit breaker" (fact)
    140: "NOISE_MEME",      # "wife's boyfriend told me to carve"
    141: "NOISE_MEME",      # "Fidelity charts are the worst"
    142: "NOISE_MEME",      # poker app spam
    143: "FINANCIAL_CLAIM", # Toyota Tundra autonomous ghostriding (fact)
    144: "ANALYSIS_DD",     # Sysco thesis: restaurants fail, customer loss est
    145: "FINANCIAL_CLAIM", # "$300 to $15k on FSLY puts" trade numbers
    146: "FINANCIAL_CLAIM", # TSMC beat earnings, AAPL/AMD customers
    147: "NOISE_MEME",      # Nikola "real story" satire/trolling
    148: "NOISE_MEME",      # "just here for money" meme
    149: "NOISE_MEME",      # "what stocks are in Becky" list gossip
    150: "NOISE_MEME",      # "get in boys, going to mars"
    151: "NOISE_MEME",      # "punish the shorts" hype
    152: "ANALYSIS_DD",     # EU USB-C ruling impact on Apple licensing
    153: "NOISE_MEME",      # "the sign bears have been looking for"
    154: "GENUINE_OPINION", # UPS anecdote + long position, thin reasoning
    155: "FINANCIAL_CLAIM", # Constellation buys Gary Vaynerchuk wines (fact)
    156: "NOISE_MEME",      # "buy rice" apocalypse copypasta
    157: "NOISE_MEME",      # "how to get diamond hands" process
    158: "NOISE_MEME",      # NTD "special report" link, no info
    159: "ANALYSIS_DD",     # Pawn stocks business model + thesis
    160: "NOISE_MEME",      # CDC gatherings / rip WWE
    161: "NOISE_MEME",      # "Skynet = China" vague thought
    162: "NOISE_MEME",      # "didn't get on the podium"
    163: "NOISE_MEME",      # "bears enjoying tendies"
    164: "NOISE_MEME",      # earnings thread
    165: "NOISE_MEME",      # "how fcked am I"
    166: "GENUINE_OPINION", # "MSFT how safe am I, bulls only" bullish ask
    167: "GENUINE_OPINION", # "hold the puts / dumping on red open"
    168: "GENUINE_OPINION", # "will WMT calls keep printing" bullish ask
    169: "NOISE_MEME",      # "thank you based protestors" politics
    170: "NOISE_MEME",      # "grandpa uses Robinhood"
    171: "NOISE_MEME",      # "how much have you made" survey
    172: "NOISE_MEME",      # "advanced portfolio funding" copypasta
    173: "NOISE_MEME",      # "GUH new year" copypasta
    174: "NOISE_MEME",      # "Shkrelli out of prison" news w/o market content
    175: "NOISE_MEME",      # broker question (Europa)
    176: "NOISE_MEME",      # "SPCE tendies" one-liner
    177: "NOISE_MEME",      # "bless my spy puts" reaction
    178: "NOISE_MEME",      # "earn money downloading apps" spam
    179: "NOISE_MEME",      # "show my Amazon tendies" flex
    180: "GENUINE_OPINION", # "election day (buy puts)" bearish
    181: "NOISE_MEME",      # wsbvotebot log
    182: "NOISE_MEME",      # funko meme
    183: "NOISE_MEME",      # "day spending gains" flex
    184: "NOISE_MEME",      # "bored, need news"
    185: "FINANCIAL_CLAIM", # Papa John founder eats 50 pizzas (relayed fact)
    186: "FINANCIAL_CLAIM", # Sonos-Legrand licensing deal (relayed fact)
    187: "FINANCIAL_CLAIM", # CVAC position 30 shares @ $51.86, up 18%
    188: "ANALYSIS_DD",     # TA map: weekly 200MA + .382 fib support
    189: "NOISE_MEME",      # "introduced in-law to options"
    190: "GENUINE_OPINION", # negative rate expectation, thin reasoning
    191: "FINANCIAL_CLAIM", # Playboy to go public (fact)
    192: "NOISE_MEME",      # "help me get into discord"
    193: "NOISE_MEME",      # "inverse everything" copypasta
    194: "NOISE_MEME",      # asks for bond-buying explanation, no info
    195: "NOISE_MEME",      # "OG TRIL holders" meme
    196: "NOISE_MEME",      # "an interesting title"
    197: "NOISE_MEME",      # "real loss was time spent trading" joke
    198: "FINANCIAL_CLAIM", # Pfizer vaccine positive data (news fact)
    199: "NOISE_MEME",      # "art in Brooklyn looks right"
}

ALLOWED = {"NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"}
N = 200

assert set(LABELS_BY_INDEX) == set(range(N)), (
    sorted(set(range(N)) - set(LABELS_BY_INDEX)),
    sorted(set(LABELS_BY_INDEX) - set(range(N))),
)
labels = [LABELS_BY_INDEX[i] for i in range(N)]
assert len(labels) == N
assert all(l in ALLOWED for l in labels)
assert "UNCERTAIN" not in labels

import json
with open("manual_labels/chunk_007.json", "w", encoding="utf-8") as f:
    json.dump({"labels": labels}, f, indent=1)
    f.write("\n")

from collections import Counter
print("len:", len(labels))
for k in ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]:
    print(k, Counter(labels)[k])
