import os
import requests

def download_file(url: str, filename: str):
    print(f"Downloading {filename} ... ", end="")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"FAILED ({e})")
        return

    with open(filename, "wb") as f:
        f.write(r.content)
    print("OK")

def main():
    base = "https://top500.org/lists/top500/{year}/{month}/download/TOP500_{year}{month}_all.xml"

    # 1993.06 → 2006.11
    for year in range(2025, 2026):
        for month in ("06", "11"):
            # первые годы могли иметь только июньские списки — ноябрь многие отсутствуют
            url = base.format(year=year, month=month)
            out_name = f"TOP500_{year}{month}.xml"

            download_file(url, out_name)

if __name__ == "__main__":
    main()
