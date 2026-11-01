#!/usr/bin/env python3
"""Shared data utilities for WSB semantic search.

The corpora are Reddit submissions fetched from Arctic Shift.  A post is a
title plus an optional selftext.  Long selftexts are split into overlapping
chunks so that a single chunk stays close to the encoder's useful input size.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_POSTS = Path(__file__).resolve().parent / "data" / "wsb_labeled_posts.json"

TICKER_COMPANIES: Dict[str, Tuple[str, Sequence[str]]] = {
    "AAPL": ("Apple", ("apple", "apple inc", "аппл", "эппл", "яблоко")),
    "MSFT": ("Microsoft", ("microsoft", "майкрософт")),
    "GOOGL": ("Google", ("google", "alphabet", "гугл", "алфавит")),
    "AMZN": ("Amazon", ("amazon", "амазон")),
    "TSLA": ("Tesla", ("tesla", "тесла")),
    "NVDA": ("Nvidia", ("nvidia", "нвидиа", "нвидия")),
    "META": ("Meta", ("meta platforms", "facebook", "фейсбук", "мета")),
    "AMD": ("AMD", ("advanced micro devices",)),
    "GME": ("GameStop", ("gamestop", "game stop", "геймстоп")),
    "AMC": ("AMC", ("amc entertainment", "амс")),
    "SPY": ("SPY", ("s&p 500", "s&p500", "sp500", "s and p 500", "снп 500")),
    "QQQ": ("QQQ", ("nasdaq 100", "nasdaq-100", "наздак 100")),
    "DIA": ("DIA", ("dow jones", "доу джонс")),
    "IWM": ("IWM", ("russell 2000", "расселл 2000")),
    "NFLX": ("Netflix", ("netflix", "нетфликс")),
    "DIS": ("Disney", ("disney", "walt disney", "дисней")),
    "BA": ("Boeing", ("boeing", "боинг")),
    "JPM": ("JPMorgan", ("jpmorgan", "jp morgan", "джп морган")),
    "BAC": ("Bank of America", ("bank of america", "bofa", "банк америки")),
    "WFC": ("Wells Fargo", ("wells fargo", "уэллс фарго")),
    "C": ("Citigroup", ("citigroup", "citi", "ситигрупп")),
    "GS": ("Goldman Sachs", ("goldman sachs", "goldman", "голдман сакс")),
    "MS": ("Morgan Stanley", ("morgan stanley", "морган стэнли")),
    "V": ("Visa", ("visa inc",)),
    "MA": ("Mastercard", ("mastercard", "мастеркард")),
    "PYPL": ("PayPal", ("paypal", "пейпал")),
    "SQ": ("Block", ("block inc", "square inc")),
    "COIN": ("Coinbase", ("coinbase", "коинбейс")),
    "HOOD": ("Robinhood", ("robinhood", "робинхуд")),
    "PLTR": ("Palantir", ("palantir", "палантир")),
    "SNOW": ("Snowflake", ("snowflake", "сноуфлейк")),
    "CRM": ("Salesforce", ("salesforce", "сейлсфорс")),
    "ORCL": ("Oracle", ("oracle", "оракл")),
    "INTC": ("Intel", ("intel", "интел")),
    "MU": ("Micron", ("micron", "микрон")),
    "TSM": ("TSMC", ("tsmc", "taiwan semiconductor", "тсмс")),
    "ASML": ("ASML", ("asml holding",)),
    "AVGO": ("Broadcom", ("broadcom", "бродком")),
    "QCOM": ("Qualcomm", ("qualcomm", "квалкомм")),
    "ARM": ("Arm", ("arm holdings",)),
    "SMCI": ("Super Micro", ("super micro", "supermicro", "супермикро")),
    "DELL": ("Dell", ("dell", "делл")),
    "IBM": ("IBM", ("international business machines",)),
    "UBER": ("Uber", ("uber", "убер")),
    "LYFT": ("Lyft", ("lyft", "лифт")),
    "ABNB": ("Airbnb", ("airbnb", "эйрбиэнби")),
    "DASH": ("DoorDash", ("doordash", "дурдаш")),
    "SHOP": ("Shopify", ("shopify", "шопифай")),
    "RIVN": ("Rivian", ("rivian", "ривиан")),
    "LCID": ("Lucid", ("lucid motors", "lucid group")),
    "NIO": ("Nio", ("nio inc", "нио")),
    "XPEV": ("XPeng", ("xpeng", "сяопэн")),
    "LI": ("Li Auto", ("li auto", "ли ауто")),
    "F": ("Ford", ("ford motor",)),
    "GM": ("General Motors", ("general motors", "дженерал моторс")),
    "RACE": ("Ferrari", ("ferrari", "феррари")),
    "TM": ("Toyota", ("toyota", "тойота")),
    "HMC": ("Honda", ("honda", "хонда")),
    "BRK.B": ("Berkshire Hathaway", ("berkshire hathaway", "berkshire", "brk.b", "brkb", "беркшир")),
    "BABA": ("Alibaba", ("alibaba", "алибаба")),
    "JD": ("JD.com", ("jd.com",)),
    "PDD": ("PDD", ("pinduoduo", "temu", "пиндуодуо")),
    "BIDU": ("Baidu", ("baidu", "байду")),
    "SONY": ("Sony", ("sony", "сони")),
    "SONO": ("Sonos", ("sonos", "сонос")),
    "ROKU": ("Roku", ("roku", "року")),
    "SPOT": ("Spotify", ("spotify", "спотифай")),
    "SNAP": ("Snap", ("snap inc", "snapchat", "снэпчат")),
    "PINS": ("Pinterest", ("pinterest", "пинтерест")),
    "RDDT": ("Reddit", ("reddit", "реддит")),
    "MSTR": ("MicroStrategy", ("microstrategy", "micro strategy", "микростратеги")),
    "MARA": ("Marathon Digital", ("marathon digital", "marathon patent")),
    "RIOT": ("Riot Platforms", ("riot platforms", "riot blockchain")),
    "BTC": ("Bitcoin", ("bitcoin", "биткоин", "биткойн")),
    "ETH": ("Ethereum", ("ethereum", "эфириум", "эфир")),
    "DOGE": ("Dogecoin", ("dogecoin", "догикоин")),
    "GLD": ("Gold", ("gold etf", "золото")),
    "SLV": ("Silver", ("silver etf", "серебро")),
    "XLE": ("Energy Select", ("energy select sector",)),
    "XOM": ("Exxon", ("exxon", "exxonmobil", "эксон")),
    "CVX": ("Chevron", ("chevron", "шеврон")),
    "OXY": ("Occidental", ("occidental petroleum", "оксидентал")),
    "SLB": ("Schlumberger", ("schlumberger", "шлюмберже")),
    "LMT": ("Lockheed Martin", ("lockheed martin", "локхид мартин")),
    "RTX": ("Raytheon", ("raytheon", "рэйтеон")),
    "NOC": ("Northrop Grumman", ("northrop grumman",)),
    "GD": ("General Dynamics", ("general dynamics",)),
    "PFE": ("Pfizer", ("pfizer", "пфайзер")),
    "MRNA": ("Moderna", ("moderna", "модерна")),
    "JNJ": ("Johnson & Johnson", ("johnson & johnson", "johnson and johnson")),
    "LLY": ("Eli Lilly", ("eli lilly", "лилли")),
    "UNH": ("UnitedHealth", ("unitedhealth", "united health")),
    "CVS": ("CVS", ("cvs health",)),
    "WBA": ("Walgreens", ("walgreens", "уолгринс")),
    "KO": ("Coca-Cola", ("coca cola", "coca-cola", "кока кола")),
    "PEP": ("Pepsi", ("pepsi", "pepsico", "пепси")),
    "MCD": ("McDonald's", ("mcdonalds", "mcdonald's", "макдональдс")),
    "SBUX": ("Starbucks", ("starbucks", "старбакс")),
    "NKE": ("Nike", ("nike", "найк")),
    "LULU": ("Lululemon", ("lululemon", "лулулемон")),
    "TGT": ("Target", ("target corp", "target corporation")),
    "WMT": ("Walmart", ("walmart", "волмарт")),
    "COST": ("Costco", ("costco", "костко")),
    "HD": ("Home Depot", ("home depot", "хоум депо")),
    "LOW": ("Lowe's", ("lowes", "lowe's")),
    "PG": ("Procter & Gamble", ("procter & gamble", "procter and gamble")),
    "T": ("AT&T", ("at&t", "at and t")),
    "VZ": ("Verizon", ("verizon", "веризон")),
    "TMUS": ("T-Mobile", ("t-mobile", "t mobile")),
    "CMCSA": ("Comcast", ("comcast", "комкаст")),
    "CHTR": ("Charter", ("charter communications",)),
    "PARA": ("Paramount", ("paramount", "парамаунт")),
    "WBD": ("Warner Bros", ("warner bros", "warner brothers")),
    "SPCE": ("Virgin Galactic", ("virgin galactic", "верджин галактик")),
    "ASTS": ("AST SpaceMobile", ("ast spacemobile", "spacemobile")),
    "SANA": ("Sana Biotechnology", ("sana biotechnology",)),
    "SOFI": ("SoFi", ("sofi", "софи")),
    "AFRM": ("Affirm", ("affirm", "аффирм")),
    "UPST": ("Upstart", ("upstart", "апстарт")),
    "BBAI": ("BigBear.ai", ("bigbear.ai", "bigbear")),
    "IONQ": ("IonQ", ("ionq", "ионк")),
    "RGTI": ("Rigetti", ("rigetti", "ригетти")),
    "OKLO": ("Oklo", ("oklo", "окло")),
    "SMR": ("NuScale", ("nuscale", "нускейл")),
    "CEG": ("Constellation Energy", ("constellation energy",)),
    "VST": ("Vistra", ("vistra", "вистра")),
    "NRG": ("NRG", ("nrg energy",)),
    "ENPH": ("Enphase", ("enphase", "энфейз")),
    "FSLR": ("First Solar", ("first solar",)),
    "RUN": ("Sunrun", ("sunrun", "санран")),
    "PLUG": ("Plug Power", ("plug power",)),
    "BE": ("Bloom Energy", ("bloom energy",)),
    "QS": ("QuantumScape", ("quantumscape", "квантумскейп")),
    "CHPT": ("ChargePoint", ("chargepoint", "чарджпоинт")),
    "EVGO": ("EVgo", ("evgo",)),
    "NEE": ("NextEra", ("nextera", "некстера")),
}

TICKER_ALIASES: Dict[str, str] = {}
for _ticker, (_company, _aliases) in TICKER_COMPANIES.items():
    TICKER_ALIASES[_company.lower()] = _ticker
    for _alias in _aliases:
        TICKER_ALIASES[_alias.lower()] = _ticker

TICKER_ALIASES["appl"] = "AAPL"
TICKER_ALIASES["apple inc."] = "AAPL"
TICKER_ALIASES["apple computer"] = "AAPL"
TICKER_ALIASES.update({
    "яблока": "AAPL", "яблоки": "AAPL", "яблоком": "AAPL", "эппла": "AAPL",
    "теслы": "TSLA", "тесле": "TSLA", "теслу": "TSLA",
    "геймстопа": "GME", "геймстопу": "GME", "геймстопом": "GME",
    "нвидии": "NVDA", "нвидию": "NVDA",
    "майкрософта": "MSFT", "майкрософту": "MSFT",
    "амазона": "AMZN", "амазону": "AMZN",
    "гугла": "GOOGL", "гуглу": "GOOGL",
    "меты": "META", "мете": "META",
    "нетфликса": "NFLX", "диснея": "DIS",
})

AMBIGUOUS_COMPANY_TICKERS = {
    "C", "V", "F", "T", "MA", "HD", "LOW", "LI", "ARM", "BE", "RUN", "GD", "MS",
    "TGT", "GLD", "SLV", "RDDT", "SNAP", "SQ", "CHTR", "AFRM", "UPST",
}

SAFE_COMPANY_ALIASES = {
    alias
    for ticker, (company, aliases) in TICKER_COMPANIES.items()
    for alias in (company.lower(), *aliases)
    if ticker not in AMBIGUOUS_COMPANY_TICKERS and len(alias) >= 4
}

TICKER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{1,5}(?:\.[A-Z])?)(?![A-Za-z0-9])")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*")

NOT_TICKERS = {
    "A", "I", "AM", "PM", "DD", "CEO", "CFO", "CTO", "IPO", "ETF", "EPS", "GDP", "CPI",
    "FED", "SEC", "FDA", "IRS", "USA", "US", "UK", "EU", "AI", "ML", "IT", "HR", "PR",
    "VS", "YOLO", "FOMO", "ATH", "ATL", "OTM", "ITM", "IV", "OI", "TA", "FA", "TLDR",
    "TL", "DR", "EDIT", "IMO", "IMHO", "LOL", "WTF", "OP", "NSFW", "FYI", "PSA", "EOD",
    "AH", "PM", "RH", "WSB", "APES", "APE", "HODL", "MOASS", "NFT", "NFTs", "DCA",
    "CALL", "PUTS", "PUT", "BUY", "SELL", "HOLD", "LONG", "SHORT", "NEWS", "MEME",
    "GAIN", "LOSS", "YOLO", "SHITPOST", "DISCUSSION", "OPTIONS", "STOCKS", "CHART",
    "TECHNICAL", "FUNDAMENTALS", "EARNINGS", "THREAD", "DAILY", "SATIRE", "STORYTIME",
    "FUTURES", "FOREX", "CRYPTO", "BOND", "BONDS", "CASH", "USD", "EUR", "GBP", "JPY",
    "CNY", "RUB", "ETH", "BTC", "DOGE", "SPX", "NDX", "DJI", "VIX", "DXY", "GLD",
    "SLV", "USO", "TQQQ", "SQQQ", "UVXY", "TVIX", "SOXL", "SOXS", "ARKK", "ARKG",
    "IWM", "DIA", "QQQ", "SPY", "VOO", "VTI", "VXUS", "BND", "AGG", "LQD", "HYG",
}

RUSSIAN_QUERY_WORDS = {
    "акции": "stock",
    "акция": "stock",
    "акций": "stock",
    "акциях": "stock",
    "акциями": "stock",
    "акцию": "stock",
    "бумаги": "stock",
    "бумага": "stock",
    "покупать": "buy",
    "купить": "buy",
    "покупка": "buy",
    "продавать": "sell",
    "продать": "sell",
    "продажа": "sell",
    "держать": "hold",
    "прибыль": "earnings",
    "прибыли": "earnings",
    "выручка": "revenue",
    "выручки": "revenue",
    "доход": "earnings",
    "опционы": "options",
    "опцион": "option",
    "коллы": "calls",
    "путы": "puts",
    "дивиденды": "dividend",
    "дивиденд": "dividend",
    "анализ": "analysis",
    "мнение": "opinion",
    "новости": "news",
    "новость": "news",
    "что": "what",
    "думают": "think",
    "думает": "think",
    "говорят": "say",
    "стоит": "should",
    "ли": "i",
    "почему": "why",
    "прогноз": "forecast",
    "цена": "price",
    "цель": "target",
    "график": "chart",
    "технический": "technical",
    "фундаментальный": "fundamental",
    "краткосрочный": "short term",
    "долгосрочный": "long term",
    "рост": "growth",
    "падение": "drop",
    "обвал": "crash",
    "бычий": "bullish",
    "медвежий": "bearish",
    "рынок": "market",
    "инвестор": "investor",
    "инвестиции": "investing",
    "портфель": "portfolio",
    "заработок": "profit",
    "убыток": "loss",
    "сделка": "trade",
    "трейдинг": "trading",
    "эппл": "apple",
    "тесла": "tesla",
    "геймстоп": "gamestop",
    "нвидиа": "nvidia",
    "нвидия": "nvidia",
    "майкрософт": "microsoft",
    "амазон": "amazon",
    "гугл": "google",
    "мета": "meta",
    "нетфликс": "netflix",
    "дисней": "disney",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    post_id: str
    title: str
    text: str
    body: str
    post_body: str
    chunk_index: int
    n_chunks: int
    year: str
    created_utc: int
    author: str
    score: int
    num_comments: int
    flair: str
    label: str
    tickers: Tuple[str, ...]


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[(removed|deleted)\]", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_body(body: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    if not body:
        return [""]
    if len(body) <= max_chars:
        return [body]
    sentences = split_sentences(body)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = word
                else:
                    current = f"{current} {word}".strip()
            continue
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks or [body[:max_chars]]


def detect_tickers(text: str, include_safe_company_names: bool = True) -> List[str]:
    found: List[str] = []
    seen = set()

    for match in TICKER_PATTERN.finditer(text or ""):
        token = match.group(1)
        if token in NOT_TICKERS:
            continue
        if token in TICKER_COMPANIES and token not in seen:
            seen.add(token)
            found.append(token)
            continue
        if token == "APPL" and "AAPL" not in seen:
            seen.add("AAPL")
            found.append("AAPL")

    if include_safe_company_names:
        lowered = (text or "").lower().replace("’", "'")
        for alias in SAFE_COMPANY_ALIASES:
            if len(alias) < 4:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                ticker = TICKER_ALIASES.get(alias)
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    found.append(ticker)
    return found


def canonical_alias_map() -> Dict[str, str]:
    return dict(TICKER_ALIASES)


def normalize_query(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return ""
    text = text.replace("’", "'")
    lowered = text.lower()

    replacements = []
    for alias, ticker in sorted(TICKER_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if len(alias) < 3 and alias not in {"c", "v", "f", "t"}:
            continue
        pattern = rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
        if re.search(pattern, lowered):
            company = TICKER_COMPANIES[ticker][0]
            replacements.append(f"{ticker} {company}")

    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9&.'$-]+", lowered)
    translated = [RUSSIAN_QUERY_WORDS.get(word, word) for word in words]

    parts = [" ".join(translated)]
    parts.extend(replacements)
    normalized = " ".join(p for p in parts if p.strip())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or text


def parse_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def load_posts(path: Path = DEFAULT_POSTS) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "years" in data:
        years = data["years"]
        rows = []
        for year in sorted(years):
            for post in years[year]:
                post = dict(post)
                post.setdefault("year", str(year))
                rows.append(post)
        return rows
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognised post file format: {path}")


def post_to_chunks(post: dict, max_chars: int = 700, overlap: int = 100) -> List[Chunk]:
    post_id = str(post.get("id") or post.get("post_id") or "")
    title = clean_text(post.get("title"))
    body = clean_text(post.get("body") or post.get("selftext") or "")
    created_utc = parse_int(post.get("created_utc"))
    year = str(post.get("year") or "")
    if not year and created_utc:
        from datetime import datetime, timezone
        year = str(datetime.fromtimestamp(created_utc, timezone.utc).year)
    author = str(post.get("author") or "")
    score = parse_int(post.get("score"))
    num_comments = parse_int(post.get("num_comments"))
    flair = clean_text(post.get("flair") or post.get("link_flair_text"))
    label = str(post.get("label") or "")
    tickers = tuple(detect_tickers(f"{title}\n{body}"))
    body_chunks = chunk_body(body, max_chars=max_chars, overlap=overlap)
    chunks: List[Chunk] = []
    for index, body_chunk in enumerate(body_chunks):
        if body_chunk:
            text = f"{title}\n\n{body_chunk}".strip() if title else body_chunk
        else:
            text = title
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{post_id}:{index}",
            post_id=post_id,
            title=title,
            text=text,
            body=body_chunk,
            post_body=body,
            chunk_index=index,
            n_chunks=len(body_chunks),
            year=year,
            created_utc=created_utc,
            author=author,
            score=score,
            num_comments=num_comments,
            flair=flair,
            label=label,
            tickers=tickers,
        ))
    return chunks


def iter_chunks(posts: Iterable[dict], **kwargs) -> Iterator[Chunk]:
    for post in posts:
        yield from post_to_chunks(post, **kwargs)


def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "post_id": chunk.post_id,
        "title": chunk.title,
        "text": chunk.text,
        "body": chunk.body,
        "post_body": chunk.post_body,
        "chunk_index": chunk.chunk_index,
        "n_chunks": chunk.n_chunks,
        "year": chunk.year,
        "created_utc": chunk.created_utc,
        "author": chunk.author,
        "score": chunk.score,
        "num_comments": chunk.num_comments,
        "flair": chunk.flair,
        "label": chunk.label,
        "tickers": list(chunk.tickers),
    }


def split_posts(posts: Sequence[dict], test_size: float = 0.1, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(posts)))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(posts) * test_size)))
    test_idx = set(indices[:n_test])
    train = [p for i, p in enumerate(posts) if i not in test_idx]
    test = [p for i, p in enumerate(posts) if i in test_idx]
    return train, test


ENGLISH_TEMPLATES = (
    "{ticker} stock",
    "what does WSB think about {ticker}",
    "{company} stock discussion",
    "is {ticker} a good buy",
    "bull case for {ticker}",
    "bear case for {ticker}",
    "WSB opinion on {company}",
)

RUSSIAN_TEMPLATES = (
    "акции {ticker}",
    "что думают о {ticker}",
    "мнение о {company}",
    "стоит ли покупать {ticker}",
    "анализ {ticker}",
    "новости {company}",
    "прогноз по {ticker}",
)


def build_training_pairs(train_posts: Sequence[dict], seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    chunks_by_post: Dict[str, List[Chunk]] = {}
    for post in train_posts:
        chunks = post_to_chunks(post)
        if chunks:
            chunks_by_post[chunks[0].post_id] = chunks

    all_chunks = [c for chunks in chunks_by_post.values() for c in chunks]
    by_ticker: Dict[str, List[Chunk]] = {}
    for chunk in all_chunks:
        for ticker in chunk.tickers:
            by_ticker.setdefault(ticker, []).append(chunk)

    pairs: List[dict] = []
    for post in train_posts:
        post_id = str(post.get("id") or post.get("post_id") or "")
        chunks = chunks_by_post.get(post_id)
        if not chunks:
            continue
        positive = chunks[0]
        title = clean_text(post.get("title"))
        tickers = list(positive.tickers)

        if title and len(title) >= 18 and len(positive.body) >= 80:
            pairs.append({
                "query": title,
                "positive_id": positive.chunk_id,
                "group": f"post:{post_id}",
                "kind": "title",
                "ticker": tickers[0] if tickers else "",
            })

        for ticker in tickers[:3]:
            company = TICKER_COMPANIES.get(ticker, (ticker, ()))[0]
            english = rng.choice(ENGLISH_TEMPLATES).format(ticker=ticker, company=company)
            russian = rng.choice(RUSSIAN_TEMPLATES).format(ticker=ticker, company=company)
            for query in (english, russian):
                pairs.append({
                    "query": query,
                    "positive_id": positive.chunk_id,
                    "group": f"ticker:{ticker}",
                    "kind": "ticker",
                    "ticker": ticker,
                })

    if not pairs:
        raise ValueError("no training pairs were generated")

    for pair in pairs:
        positive = next(c for c in chunks_by_post[pair["positive_id"].split(":")[0]]
                        if c.chunk_id == pair["positive_id"])
        ticker = pair.get("ticker") or ""
        candidates = [
            c for c in all_chunks
            if c.post_id != positive.post_id and (not ticker or ticker not in c.tickers)
        ]
        hard = [c for c in candidates if positive.flair and c.flair == positive.flair]
        negative_pool = hard if len(hard) >= 2 else candidates
        rng.shuffle(negative_pool)
        pair["negative_ids"] = [c.chunk_id for c in negative_pool[:2]]
    return pairs


def build_eval_queries(test_posts: Sequence[dict], max_ticker_queries: int = 80, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    by_ticker: Dict[str, List[str]] = {}
    title_queries: List[dict] = []
    for post in test_posts:
        post_id = str(post.get("id") or post.get("post_id") or "")
        chunks = post_to_chunks(post)
        if not chunks:
            continue
        title = clean_text(post.get("title"))
        body = clean_text(post.get("body") or post.get("selftext") or "")
        for ticker in chunks[0].tickers:
            by_ticker.setdefault(ticker, []).append(post_id)
        if title and len(title) >= 20 and len(body) >= 120:
            title_queries.append({
                "query": title,
                "relevant_post_ids": [post_id],
                "kind": "title",
                "ticker": chunks[0].tickers[0] if chunks[0].tickers else "",
            })

    ticker_queries: List[dict] = []
    tickers = [t for t, posts in by_ticker.items() if len(posts) >= 3]
    rng.shuffle(tickers)
    for ticker in tickers[:max_ticker_queries]:
        company = TICKER_COMPANIES.get(ticker, (ticker, ()))[0]
        for template in ("акции {ticker}", "what does WSB think about {ticker}", "{company} stock"):
            ticker_queries.append({
                "query": template.format(ticker=ticker, company=company),
                "relevant_post_ids": sorted(set(by_ticker[ticker])),
                "kind": "ticker",
                "ticker": ticker,
            })
    rng.shuffle(title_queries)
    return ticker_queries + title_queries[: max(40, len(ticker_queries) // 3)]
