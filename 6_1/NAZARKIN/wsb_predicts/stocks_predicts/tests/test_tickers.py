"""Ticker extraction tests (cashtags, aliases, ambiguity gate)."""
from __future__ import annotations

from extract_tickers import extract_tickers, load_ticker_config, make_fragment


def _cfg():
    return load_ticker_config("configs/tickers.yaml")


def test_cashtags_and_aliases_are_linked():
    got = {r["ticker"] for r in extract_tickers("$TSLA calls and Nvidia puts", _cfg())}
    assert {"TSLA", "NVDA"}.issubset(got)


def test_ambiguous_and_stoplisted_tokens_are_rejected():
    assert extract_tickers("DD on my AI play, CEO said IT is ON", _cfg()) == []


def test_universe_filter_rejects_unknown_symbols():
    assert extract_tickers("$FAKE and $ZZZZ", _cfg()) == []


def test_fragment_is_bounded_and_normalised():
    text = "x" * 500 + " $GME squeeze " + "y" * 500
    frag = make_fragment(text, 510, 40)
    assert "GME" in frag
    assert len(frag) <= 90
    assert "\n" not in frag


def test_max_tickers_per_post_is_respected():
    cfg = _cfg()
    text = "$AAPL $TSLA $NVDA $MSFT $AMZN $META $AMD $NFLX"
    got = extract_tickers(text, cfg)
    assert len(got) <= int(cfg.get("max_per_post", 5))
