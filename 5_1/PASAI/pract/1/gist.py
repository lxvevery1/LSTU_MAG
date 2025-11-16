#!/usr/bin/env python3
"""
TOP500 Energy Efficiency Analysis for ratings 31, 32, 33 (2008-2009)

Dependencies:
    pip install requests lxml matplotlib pandas numpy scipy
"""

import os
import requests
from lxml import etree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tabulate import tabulate

# ---------- CONFIG ----------
RANK_TO_DATE = {
    33: (2009, 6),
    32: (2008, 11),
    31: (2008, 6),
}
BASE_URL = "https://top500.org/lists/top500/{year}/{month:02d}/download/TOP500_{year}{month:02d}_all.xml"
OUT_XML = "xml"
OUT_CSV = "top500_efficiency.csv"
OUT_PLOTS = "top500_efficiency_plots"
TIMEOUT = 20
# ----------------------------

os.makedirs(OUT_XML, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# ---------- Download ----------
def download_xml(year, month):
    url = BASE_URL.format(year=year, month=month)
    fname = os.path.join(OUT_XML, f"TOP500_{year}{month:02d}.xml")
    if os.path.exists(fname):
        print(f"[SKIP] {fname} already exists")
        return fname, True
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            with open(fname, "wb") as f:
                f.write(r.content)
            print(f"[OK]   {fname} downloaded")
            return fname, True
        else:
            print(f"[MISS] {url} ({r.status_code})")
            return None, False
    except Exception as e:
        print(f"[ERR]  {url} -> {e}")
        return None, False

# ---------- Parse XML ----------
def parse_xml(path):
    """Return list of dicts with system, rank, power, rmax, efficiency"""
    with open(path, "rb") as f:
        xml_bytes = f.read()
    root = etree.fromstring(xml_bytes)
    ns_uri = root.nsmap.get("top500")
    if ns_uri is None:
        print(f"[WARN] Namespace not found in {path}")
        return []
    ns = {"top500": ns_uri}

    records = []
    for site in root.findall("top500:site", namespaces=ns):
        try:
            name = site.findtext("top500:system-name", namespaces=ns)
            if not name:
                name = site.findtext("top500:computer", namespaces=ns)
            rank = site.findtext("top500:rank", namespaces=ns)
            power = site.findtext("top500:power", namespaces=ns)
            rmax = site.findtext("top500:r-max", namespaces=ns)

            if name and rank and power and rmax:
                p = float(power)
                r = float(rmax)
                if p > 0 and r > 0:
                    records.append({
                        "system": name.strip(),
                        "rank": int(rank),
                        "power": p,
                        "rmax": r,
                        "efficiency": r / p  # FLOPS per kW
                    })
        except Exception:
            continue
    return records

# ---------- Plot Histogram + KDE ----------
def plot_hist(df, label):
    arr = df["efficiency"].values
    fig, ax1 = plt.subplots(figsize=(8,5))

    counts, bins, _ = ax1.hist(
        arr,
        bins="auto",
        alpha=0.6,
        color="steelblue",
        edgecolor="black"
    )

    ax1.set_xticks(bins)
    ax1.set_xticklabels([f"{b:.0f}" for b in bins], rotation=45)

    ax1.set_xlabel("Energy Efficiency (Rmax / Power, FLOPS/kW)")
    ax1.set_ylabel("Count", color="steelblue")
    ax1.tick_params(axis='y', labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.hist(arr, bins=bins, density=True, alpha=0.0)
    kde = gaussian_kde(arr)
    xs = np.linspace(min(arr), max(arr), 200)
    ax2.plot(xs, kde(xs), color="red", lw=2, label="KDE")
    ax2.set_ylabel("Density", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title(f"TOP500 Energy Efficiency — {label} (N={len(arr)})")
    ax2.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    fname = os.path.join(OUT_PLOTS, f"top500_efficiency_{label}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[PLOT] {fname} saved")

# ---------- Summary plot ----------
def plot_summary(df):
    df_valid = df[(df["efficiency"].notna()) & (df["efficiency"] > 0)]
    stats = df_valid.groupby("list_rank")["efficiency"].agg(["min", "mean", "max"])
    labels = [f"List#{r}" for r in stats.index.values]

    plt.figure(figsize=(10,6))
    width = 0.25
    x = np.arange(len(labels))
    plt.bar(x - width, stats["min"], width, label="Min", color="skyblue")
    plt.bar(x, stats["mean"], width, label="Mean", color="orange")
    plt.bar(x + width, stats["max"], width, label="Max", color="green")

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Energy Efficiency (FLOPS/kW)")
    plt.title("TOP500 Energy Efficiency – Min / Mean / Max per List")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    fname = os.path.join(OUT_PLOTS, "top500_efficiency_summary.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[PLOT] {fname} saved")

# ---------- CDF -----------
def plot_cdf(df, label):
    """Plot empirical cumulative distribution function (CDF) for efficiency."""
    arr = np.sort(df["efficiency"].values)
    cdf = np.arange(1, len(arr) + 1) / len(arr)

    plt.figure(figsize=(8,5))
    plt.plot(arr, cdf, color="darkgreen", lw=2)
    plt.xlabel("Energy Efficiency (Rmax / Power, FLOPS/kW)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of Energy Efficiency — {label}")
    plt.grid(True, linestyle=":", alpha=0.5)

    # X ticks — реальные значения энергоэффективности
    xticks = np.linspace(min(arr), max(arr), 10)
    plt.xticks(xticks, [f"{x:.0f}" for x in xticks], rotation=45)

    plt.tight_layout()
    fname = os.path.join(OUT_PLOTS, f"top500_efficiency_cdf_{label}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[CDF]  {fname} saved")

# ---------- Main ----------
def main():
    all_records = []
    for rank, (year, month) in RANK_TO_DATE.items():
        path, ok = download_xml(year, month)
        if not ok:
            continue
        recs = parse_xml(path)
        print(f"Rank {rank}, year {year}, month {month}: parsed {len(recs)} records")
        for r in recs:
            r["year"] = year
            r["month"] = month
            r["list_rank"] = rank
        all_records.extend(recs)

    if len(all_records) == 0:
        print("[ERROR] No data collected! Check XML parsing.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False)
    print(f"[CSV] Saved {OUT_CSV} ({len(df)} rows)")

    # Histogram per list
    for rank in sorted(df["list_rank"].unique()):
        df_rank = df[df["list_rank"] == rank]
        if len(df_rank) > 0:
            plot_hist(df_rank, f"List#{rank}")
            plot_cdf(df_rank, f"List#{rank}")

    # Summary across lists
    plot_summary(df)

    # ---------- Display table ----------
    print("\n=== TOP500 Energy Efficiency Table ===")
    table = df[["list_rank", "rank", "system", "rmax", "power", "efficiency"]]
    table_sorted = table.sort_values(["list_rank", "rank"]).reset_index(drop=True)
    table_sorted_eff = table_sorted.sort_values("efficiency", ascending=False)
    print(tabulate(table_sorted_eff.head(20), headers="keys", tablefmt="github", floatfmt=".3f"))
    print("... (only first 20 rows shown)\n")

if __name__ == "__main__":
    main()
