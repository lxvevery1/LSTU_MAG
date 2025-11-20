#!/usr/bin/env python3
"""
Variant #17
Correlation:
    X = HPL Rmax
    Y = HPL Efficiency = Rmax / Rpeak

For lists:
    32 → 2008.11
    33 → 2009.06
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
from tabulate import tabulate

XML_DIR = "xml"

LISTS = {
    31: (2007, 11),
    56: (2021, 6),
}

OUT_PLOT = "variant17_scatter.png"


def load_xml(year, month):
    fname = f"TOP500_{year}{month:02d}.xml"
    path = os.path.join(XML_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERR] missing XML file: {path}")
        return None
    with open(path, "rb") as f:
        return etree.fromstring(f.read())


def parse_list(root, list_id):
    ns = {"top500": root.nsmap.get("top500")}

    rows = []
    for site in root.findall("top500:site", namespaces=ns):
        try:
            rmax = float(site.findtext("top500:r-max", namespaces=ns))
            rpeak = float(site.findtext("top500:r-peak", namespaces=ns))

            if rmax <= 0 or rpeak <= 0:
                continue

            eff = rmax / rpeak

            rows.append({
                "list": list_id,
                "rmax": rmax,
                "rpeak": rpeak,
                "eff_hpl": eff
            })
        except:
            continue

    return rows


def scatter(df, label, ax):
    x = df["rmax"]
    y = df["eff_hpl"]

    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("HPL Rmax")
    ax.set_ylabel("HPL Efficiency (Rmax / Rpeak)")
    ax.set_title(f"Scatter — {label}")

    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f"Pearson: {corr:.3f}", transform=ax.transAxes, va="top")

    return corr


def main():
    all_rows = []

    for list_id, (yy, mm) in LISTS.items():
        root = load_xml(yy, mm)
        if root is None:
            continue
        rows = parse_list(root, list_id)
        print(f"[OK] List {list_id} parsed: {len(rows)} rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print("\n=== Parsed data ===")
    print(tabulate(df.head(10), headers="keys", tablefmt="github", floatfmt=".3f"))

    df32 = df[df["list"] == 31]
    df33 = df[df["list"] == 56]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    c32 = scatter(df32, "List #32 (2008.11)", ax1)
    c33 = scatter(df33, "List #33 (2009.06)", ax2)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()

    print(f"\n[PLOT] saved as {OUT_PLOT}")

    print("\n=== Correlations ===")
    print(f"2008.11 (List 55): {c32:.4f}")
    print(f"2009.06 (List 56): {c33:.4f}")


if __name__ == "__main__":
    main()
