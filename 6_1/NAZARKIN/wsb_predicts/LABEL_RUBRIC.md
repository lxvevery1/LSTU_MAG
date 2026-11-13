# WSB post labelling rubric (read fully before labelling)

You are hand-labelling WallStreetBets posts. Every post is read one by one and
assigned exactly ONE class. There are only FOUR valid classes. `UNCERTAIN` is
**forbidden** - it must never appear in your output. When in doubt you still must
pick the best of the four (see tie-break rules at the bottom).

## The four classes

### 1. NOISE_MEME
The post carries no meaningful financial/investment information.
- pure jokes, shitposts, memes, reaction images, "image-style captions", copypasta
- greetings, banter, greetings-for-upvotes, "thanks stranger", awards, mod/meta posts
- questions about the *process* of investing rather than any investment content
  ("what broker do you use", "how do I get into options", "how do you guys
  survive losses", "what's your portfolio", "how did you get started")
- daily/weekly discussion thread posts, weekend threads, gain/loss picture posts
  whose caption carries no info ("up 200% on my first yolo"), pure brag posts
- link/image posts with no written argument ("chart", "news flash", a bare URL)
- off-topic life talk, politics without market content, sports, memes about
  anything, "first time posting", NSFW, greetings
- posts that are only an emoji / one-liner reaction ("up", "down", "rip", "🙂")

### 2. GENUINE_OPINION
A genuine bullish/bearish/trading **stance or intention**, with little or no
substantive analysis or evidence behind it.
- directional calls: "GME to the moon", "puts printing", "AMD is going to zero",
  "im bullish on SPY into the weekend", "selling my NVDA here"
- intended action: "yoloing 5k into QQQ calls", "adding to my RIVN position",
  "going all in on the inverse ETF", "cashing out at 150", "holding bags"
- personal trading narrative with an opinion but no numbers/reasoning:
  "I bought at 90 and it dumped, lesson learned"
- sentiment-only posts about a ticker, even short ones ("$AMD ugh", "RIP $GME")
- questions that *are* about a specific stock's future framed as a view
  ("you think NVDA clears 150 this month?")

### 3. FINANCIAL_CLAIM
A **concrete, checkable factual statement** about a company, market, event, price,
earnings, financial metric, macro data, filings, product news, or the poster's own
verified position numbers - asserted as fact, without real reasoning/valuation.
- reported numbers: "Q2 revenue was $81.8B, up 5% YoY", "SPX closed at 2495",
  "CPI came in at 3.5%", "unemployment hit a 3-year high"
- event facts: "the Fed cut rates 50bps on Wednesday", "Elon announced the
  Cybercab for Oct 10", "AAPL split 4-for-1 on Aug 31", "GameStop's next
  earnings call is March 26"
- relayed news with a specific fact: "SEC 13F shows hedge funds loaded up on TSLA"
- corporate/financial specifics: "their debt matures in 2027", "short interest is
  22% of float", "market cap is only $1.2B", "they beat on margins"
- concrete position/trade facts without analysis: "I sold 200 shares at 187, was
  down 3k", "my position: 50 calls, expiry friday"
- factual company description / business-model questions with substance

### 4. ANALYSIS_DD
**Substantive reasoning**: calculations, valuation, model, multi-step argument,
bull/bear case, catalysts, risks, structured breakdown, or cited sources used to
make a case. This needs *reasoning*, not just length.
- "At $1.2B market cap and $400M FCF that's a 3x FCF yield, downside to $X"
- multi-step bear thesis: "Here's why I think NVDA is overvalued: revenue split,
  margins, datacenter concentration risk..."
- explicit bull case / bear case / catalyst list / risk list / timeline
- earnings analysis with numbers derived or compared, guidance interpretation
- valuation work: DCF, multiples, comparables, sum-of-parts, technical levels
  with reasons, "support at 189, resistance 200 because..."
- long posts that lay out an actual chain of reasoning even without a formal
  framework, or posts that cite/link sources and synthesise them into an argument
- deep-dive threads ("Thread: why I'm long $XYZ"), "here's my process", "my
  thesis on the whole sector", options-strategy breakdowns with math

## Tie-break rules (no UNCERTAIN allowed - decide)

1. **Meme/joke framing wins.** If the post is clearly a shitpost, reaction, or
   joke (humour is the point, even if a ticker is mentioned), label
   NOISE_MEME - even if it mentions a stock. If humour is merely the wrapper
   around real content, label by the real content.
2. **No real financial content at all -> NOISE_MEME.** Any post that is a
   question, a reaction, banter, or chatter without conveying info/opinion/analysis
   is NOISE_MEME. Do not stretch "opinion" to cover process questions.
3. **Claim vs opinion.** Did the post assert something that already happened or is
   objectively checkable (a number, an event, a filing, a price, an announcement)?
   -> FINANCIAL_CLAIM. Is it a stance/expectation/prediction/intent without
   evidence? -> GENUINE_OPINION.
4. **Reasoning present -> ANALYSIS_DD.** If the post contains real analysis
   (numbers, causal argument, bull/bear case, valuation, catalysts, risks,
   sources) it is ANALYSIS_DD even if it also states opinions and facts; pick the
   *dominant* substance. A post with only a conclusion and no reasoning is
   GENUINE_OPINION, never ANALYSIS_DD.
5. **Length is not the criterion, reasoning is.** A 5-word post can be
   GENUINE_OPINION; a 400-word story with no financial content is NOISE_MEME.
6. Bare ticker / one-word title with no body ("$GME", "AAPL") -> GENUINE_OPINION
   only if there is a bullish/bearish framing, otherwise NOISE_MEME. Empty body
   is common; judge from the title plus flair context.
7. Posts that quote news and ask for opinions ("thoughts?", "what does this mean
   for the stock?") while restating a concrete fact -> FINANCIAL_CLAIM.
8. Award/announcement posts from mods and the bot -> NOISE_MEME.
9. Genuine "storytime" trade write-ups: if the emphasis is the lesson/analysis of
   *why* (numbers, mistakes, process reasoning) -> ANALYSIS_DD; if it is just
   "I yoloed and it worked" with no reasoning -> GENUINE_OPINION; if it is a pure
   show-off / meme of the P&L -> NOISE_MEME.

## Output format - follow exactly

Return a JSON object of the form:

{"labels": ["NOISE_MEME", "FINANCIAL_CLAIM", ...]}

- exactly one label per post, in the same order as the input
- exactly one of: NOISE_MEME, GENUINE_OPINION, FINANCIAL_CLAIM, ANALYSIS_DD
- no UNCERTAIN, no explanations, no extra keys
- read every post, label every post; do not skip, do not repeat, do not
  stop early, do not bulk-assign a whole chunk the same class by default.
