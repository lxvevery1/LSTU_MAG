#!/usr/bin/env python3
"""
TOP500 InfiniBand Interconnect Trend Analysis (2007–2025)

Dependencies:
    pip install lxml matplotlib pandas
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree

# ---------- CONFIG ----------
XML_DIR = "xml"
OUT_CSV = "infiniband_trend.csv"
OUT_PLOT = "infiniband_trend.png"
# ----------------------------

def parse_xml(path):
    """Return number of systems mentioning InfiniBand in <computer> field."""
    with open(path, "rb") as f:
        xml_bytes = f.read()

    try:
        root = etree.fromstring(xml_bytes)
    except Exception as e:
        print(f"[ERR]  Cannot parse {path}: {e}")
        return 0, 0

    ns_uri = root.nsmap.get("top500")
    ns = {"top500": ns_uri} if ns_uri else {}

    total, infiniband = 0, 0

    for site in root.findall("top500:site", namespaces=ns) if ns_uri else root.findall("site"):
        total += 1
        comp_text = site.findtext("top500:computer", namespaces=ns) if ns_uri else site.findtext("computer")
        if comp_text and re.search(r"infini\s*band", comp_text, re.IGNORECASE):
            infiniband += 1

    return total, infiniband


def parse_all():
    records = []
    for fname in sorted(os.listdir(XML_DIR)):
        if not fname.endswith(".xml"):
            continue
        match = re.search(r"TOP500_(\d{4})(\d{2})\.xml", fname)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        path = os.path.join(XML_DIR, fname)
        total, infiniband = parse_xml(path)
        if total == 0:
            continue
        ratio = infiniband / total * 100
        label = f"{year}.{month:02d}"
        records.append({
            "year": year,
            "month": month,
            "label": label,
            "total": total,
            "infiniband": infiniband,
            "percent": ratio
        })
        print(f"[OK]  {fname}: {infiniband}/{total} ({ratio:.1f}%)")

    return pd.DataFrame(records)


def plot_trend(df):
    plt.figure(figsize=(12,6))
    x = range(len(df))
    plt.bar(x, df["infiniband"], color="steelblue", edgecolor="black", alpha=0.8)
    plt.xticks(x, df["label"], rotation=45)
    plt.ylabel("Number of Systems Using InfiniBand")
    plt.title("InfiniBand Usage Trend in TOP500 (2007–2025)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()
    print(f"[PLOT] {OUT_PLOT} saved")


def main():
    if not os.path.isdir(XML_DIR):
        print(f"[ERROR] Directory '{XML_DIR}' not found")
        return

    df = parse_all()
    if df.empty:
        print("[ERROR] No valid XML files parsed.")
        return

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[CSV] {OUT_CSV} saved ({len(df)} records)")

    print(df)

    plot_trend(df)

    print("\n=== InfiniBand Usage Table ===")
    print(df[["label", "infiniband", "total", "percent"]].to_string(index=False, formatters={"percent": "{:.1f}%".format}))


if __name__ == "__main__":
    main()
