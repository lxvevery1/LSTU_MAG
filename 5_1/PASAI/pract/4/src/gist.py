#!/usr/bin/env python3

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
from tabulate import tabulate

XML_DIR = "xml"
XML_FILE = "TOP500_202511.xml"

OUT_PLOT = "accelerator_boxplot_202511.png"


def load_xml(fname):
    path = os.path.join(XML_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERR] missing XML file: {path}")
        return None
    with open(path, "rb") as f:
        return etree.fromstring(f.read())


def classify_accelerator(text):
    if not text:
        return "Other"
    t = text.upper()

    if re.search(r"NVIDIA|A100|H100|P100|V100|H200", t):
        return "NVIDIA"

    if "AMD" in t or "INSTINCT" in t:
        return "AMD"

    if "XEON PHI" in t or "INTEL" in t:
        return "Intel"

    return "Other"


def parse_list(root):
    ns = {"top500": root.nsmap.get("top500")}

    rows = []

    for site in root.findall("top500:site", namespaces=ns):
        computer = site.findtext("top500:computer", default="", namespaces=ns)
        rpeak = site.findtext("top500:r-peak", default="", namespaces=ns)

        try:
            rpeak_val = float(rpeak)
        except:
            continue

        accel = classify_accelerator(computer)

        rows.append({
            "accelerator": accel,
            "rpeak": rpeak_val
        })

    return rows


# === форматирование чисел ===
def fmt(x):
    try:
        return f"{x:,.3f}".replace(",", " ")
    except:
        return x


def main():
    root = load_xml(XML_FILE)
    if root is None:
        return

    rows = parse_list(root)
    df = pd.DataFrame(rows)

    # оставляем только три основные группы
    df = df[df["accelerator"].isin(["NVIDIA", "AMD", "Intel"])]

    print("\n=== Parsed accelerator data (first 10 rows) ===")
    print(tabulate(df.head(10).map(fmt), headers="keys", tablefmt="github"))

    # === boxplot ===
    plt.figure(figsize=(10, 6))
    df.boxplot(column="rpeak", by="accelerator")
    plt.title("Rpeak Boxplot by Accelerator (TOP500 2025-11)")
    plt.suptitle("")
    plt.ylabel("Rpeak")
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()

    print(f"\n[PLOT] saved as {OUT_PLOT}")

    # === stats ===
    stats = df.groupby("accelerator")["rpeak"].agg(
        ["count", "mean", "std", "min", "max", "median"]
    )

    print("\n=== Statistics by accelerator ===")
    print(tabulate(stats.map(fmt), headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
