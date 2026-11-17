"""Render the AAPL direction-prediction chart used in the project demo.

The model predicts a 3-class direction for a post x ticker pair, not a price
target.  This chart overlays daily aggregated predictions on AAPL candles for
the temporal test period and adds the daily bullish score P(UP) - P(DOWN).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PRED_PATH = "reports/runs/finbert_h5_seed42/model_predictions.parquet"
PRICE_PATH = "stocks_dataset/prices_long.csv"
OUT_PATH = "reports/figures/aapl_direction_predictions.png"


def main() -> None:
    pred = pd.read_parquet(PRED_PATH)
    aapl = pred[pred.ticker == "AAPL"].copy()
    aapl["t0_date"] = pd.to_datetime(aapl.t0_date)

    daily = (
        aapl.groupby("t0_date")
        .agg(
            n=("post_id", "size"),
            pD=("p_DOWN", "mean"),
            pF=("p_FLAT", "mean"),
            pU=("p_UP", "mean"),
            true=("direction_label", "first"),
        )
        .reset_index()
    )
    daily["pred"] = daily[["pD", "pF", "pU"]].idxmax(axis=1).map(
        {"pD": "DOWN", "pF": "FLAT", "pU": "UP"}
    )
    daily["score"] = daily.pU - daily.pD
    daily["hit"] = daily.pred == daily.true

    px = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    px = px[
        (px.ticker == "AAPL")
        & (px.date >= "2024-12-01")
        & (px.date <= "2026-10-16")
    ].copy()
    px = px.sort_values("date").reset_index(drop=True)
    px["mdate"] = mdates.date2num(px.date)

    px_by_date = px.set_index("date")["close"]
    daily["close"] = daily.t0_date.map(
        lambda d: float(px_by_date[px_by_date.index <= d].iloc[-1])
        if (px_by_date.index <= d).any()
        else np.nan
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cfcfcf",
            "axes.linewidth": 0.8,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )
    fig = plt.figure(figsize=(16, 9.2), facecolor="#f7f7f5")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.2, 1.0], hspace=0.08,
        left=0.065, right=0.975, top=0.88, bottom=0.09,
    )
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, ax2):
        a.set_facecolor("white")
        a.grid(True, color="#e8e8e8", linewidth=0.7)
        a.set_axisbelow(True)

    width = 0.62
    for d, o, h, l, c in zip(px.mdate, px.open, px.high, px.low, px.close):
        up = c >= o
        col = "#2e7d32" if up else "#c62828"
        ax.vlines(d, l, h, color=col, linewidth=0.55, alpha=0.9, zorder=2)
        body_h = max(abs(c - o), 0.03)
        ax.add_patch(
            Rectangle(
                (d - width / 2, min(o, c)), width, body_h,
                facecolor=col if up else "#ffffff",
                edgecolor=col, linewidth=0.45, alpha=0.95, zorder=3,
            )
        )

    up = daily[daily.pred == "UP"]
    down = daily[daily.pred == "DOWN"]
    flat = daily[daily.pred == "FLAT"]
    ax.scatter(up.t0_date, up.close, marker="^", s=95, color="#1b5e20",
               edgecolor="white", linewidth=0.9, zorder=6)
    ax.scatter(down.t0_date, down.close, marker="v", s=95, color="#b71c1c",
               edgecolor="white", linewidth=0.9, zorder=6)
    if len(flat):
        ax.scatter(flat.t0_date, flat.close, marker="o", s=70, color="#616161",
                   edgecolor="white", linewidth=0.9, zorder=6)

    last = daily.t0_date.max()
    ax.axvline(last, color="#555555", linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
    ax.annotate(
        "last test prediction", xy=(last, 0.015), xycoords=("data", "axes fraction"),
        rotation=90, va="bottom", ha="right", fontsize=8, color="#555555",
    )

    ax.set_ylabel("AAPL price, USD", fontsize=11, color="#333333")
    ax.set_title("AAPL (Apple) — daily WSB direction predictions at H=5",
                 loc="left", fontsize=19, fontweight="bold",
                 color="#1a1a1a", pad=18)
    ax.text(
        0.0, 1.015,
        "Historical test-period overlay, 2025-01-02 … 2026-09-18.  "
        "Each marker is the daily aggregate prediction for AAPL posts "
        "(argmax of mean probabilities).  The model predicts direction, not price.",
        transform=ax.transAxes, fontsize=10.2, color="#555555", va="bottom",
    )

    post_hit = float((aapl.pred == aapl.direction_label).mean())
    base_up = float((daily.true == "UP").mean())
    stats = (
        f"{len(aapl)} test posts  ·  {len(daily)} prediction dates\n"
        f"Predicted: {len(up)} UP  ·  {len(down)} DOWN  ·  {len(flat)} FLAT\n"
        f"Date-level hit rate: {daily.hit.mean():.1%}  ·  post-level: {post_hit:.1%}\n"
        f"Always-UP baseline on these dates: {base_up:.1%}"
    )
    ax.text(
        0.012, 0.985, stats, transform=ax.transAxes, va="top", ha="left",
        fontsize=9.3, color="#333333", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf5",
                  edgecolor="#ddd6c6", alpha=0.95),
    )

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1b5e20",
               markeredgecolor="white", markersize=8, label="Prediction UP"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#b71c1c",
               markeredgecolor="white", markersize=8, label="Prediction DOWN"),
        Patch(facecolor="#2e7d32", edgecolor="#2e7d32", label="AAPL up day"),
        Patch(facecolor="#ffffff", edgecolor="#c62828", label="AAPL down day"),
    ]
    if len(flat):
        handles.insert(
            2,
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#616161",
                   markeredgecolor="white", markersize=7,
                   label="Prediction FLAT"),
        )
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.82),
        frameon=True, framealpha=0.95, facecolor="white", edgecolor="#dddddd",
        fontsize=9, ncol=2,
    )

    colors = np.where(daily.score >= 0, "#2e7d32", "#c62828")
    ax2.bar(daily.t0_date, daily.score, width=2.6, color=colors, alpha=0.78, zorder=2)
    ax2.axhline(0, color="#333333", linewidth=0.9, zorder=3)
    ax2.set_ylabel("P(UP) − P(DOWN)", fontsize=10, color="#333333")
    ax2.set_xlabel("Post date (t0)", fontsize=10.5, color="#333333")
    ax2.set_ylim(-0.16, 0.24)
    ax2.text(0.012, 0.93, "Daily aggregated model score (higher = more bullish)",
             transform=ax2.transAxes, va="top", fontsize=9.3, color="#555555")

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(labelbottom=False)
    ax.set_xlim(pd.Timestamp("2024-12-15"), pd.Timestamp("2026-10-16"))

    fig.text(
        0.065, 0.018,
        "Source: reports/runs/finbert_h5_seed42/model_predictions.parquet; "
        "prices: stocks_dataset/prices_long.csv.  Research only — not investment advice.",
        fontsize=8.6, color="#777777",
    )
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
