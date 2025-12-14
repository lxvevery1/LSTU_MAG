#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import etree

XML_DIR = "xml"
TARGET = "201711"      # рейтинг
AGG = "mean"           # можно median


# === Классификаторы ===

def detect_cpu(text):
    txt = text.lower()
    if "xeon" in txt:
        return "Intel Xeon"
    if "epyc" in txt:
        return "AMD EPYC"
    if "a64fx" in txt:
        return "Fujitsu A64FX"
    if "power" in txt:
        return "IBM POWER"
    if "grace" in txt:
        return "NVIDIA GRACE"
    return "Other CPU"


def detect_accel(text):
    txt = text.lower()
    if "nvidia" in txt:
        return "NVIDIA"
    if "instinct" in txt:
        return "AMD Instinct"
    if "phi" in txt:
        return "Intel Xeon Phi"
    return "None"


def detect_hetero(a):
    return "Heterogeneous" if a != "None" else "CPU-only"


def detect_arch(text):
    t = text.lower()
    if "mpp" in t:
        return "MPP"
    if "cluster" in t:
        return "Cluster"
    return "Unknown"


def detect_inter(text):
    t = text.lower()
    if "slingshot" in t:
        return "Slingshot"
    if "infiniband" in t:
        return "InfiniBand"
    if "omni-path" in t:
        return "Omni-Path"
    if "ethernet" in t:
        return "Ethernet"
    return "Other"


# === XML загрузка ===

def load_xml(tag):
    year = tag[:4]
    month = tag[4:]
    path = f"{XML_DIR}/TOP500_{year}{month}.xml"
    with open(path, "rb") as f:
        return etree.fromstring(f.read())


# === Парсинг ===

def parse(root):
    ns = {"top500": root.nsmap.get("top500")}
    rows = []

    for s in root.findall("top500:site", namespaces=ns):
        comp = s.findtext("top500:computer", namespaces=ns) or ""
        rpeak_txt = s.findtext("top500:r-peak", namespaces=ns)
        if not rpeak_txt:
            continue

        try:
            rpeak = float(rpeak_txt)
        except:
            continue

        cpu = detect_cpu(comp)
        accel = detect_accel(comp)
        hetero = detect_hetero(accel)
        arch = detect_arch(comp)
        inter = detect_inter(comp)

        rows.append({
            "cpu": cpu,
            "accelerator": accel,
            "heterogeneity": hetero,
            "architecture": arch,
            "interconnect": inter,
            "rpeak": rpeak
        })

    return pd.DataFrame(rows)


# === MAIN ===

def main():
    root = load_xml(TARGET)
    df = parse(root)

    # Подготовка таблицы: характеристики → средний Rpeak
    df_long = pd.melt(
        df,
        id_vars=["rpeak"],
        value_vars=["cpu", "accelerator", "heterogeneity", "architecture", "interconnect"],
        var_name="category",
        value_name="value"
    )

    table = df_long.groupby(["category", "value"])["rpeak"].agg(AGG).reset_index()

    print("\n=== Таблица агрегированных значений ===\n")
    print(table)

    # Строим heatmap
    pivot_table = table.pivot(index="value", columns="category", values="rpeak")

    plt.figure(figsize=(10, 9))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5
    )
    plt.title(f"Rpeak Heatmap by Architecture — TOP500 {TARGET}")
    plt.tight_layout()
    plt.savefig(f"heatmap_{TARGET}.png", dpi=150)
    plt.close()

    print(f"\n[OK] Heatmap сохранён: heatmap_{TARGET}.png")


if __name__ == "__main__":
    main()
