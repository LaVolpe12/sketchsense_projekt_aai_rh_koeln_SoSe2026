"""
download_data.py
----------------
Downloads QuickDraw numpy_bitmap files (.npy) from Google Cloud Storage.
Each file contains up to 100,000 greyscale 28×28 drawings for one category.
"""

import json
import os
import requests

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

with open(NAMES_PATH) as f:
    CATEGORIES: list[str] = json.load(f)

BASE_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
)


def download_category(category: str, retries: int = 5) -> None:
    filename = f"{category}.npy"
    dest = os.path.join(DATA_DIR, filename)
    tmp  = dest + ".part"

    if os.path.exists(dest):
        print(f"  [skip] {filename} already exists")
        return

    url = BASE_URL + filename.replace(" ", "%20")
    print(f"  [download] {url}")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                if r.status_code == 404:
                    print(f"  [404]   {filename} — not in dataset, skipping")
                    return
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.rename(tmp, dest)   # atomic move — only on success
            break
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < retries:
                print(f"  [retry {attempt}/{retries}] {e}")
            else:
                print(f"  [fail]  {filename} after {retries} attempts: {e}")
                return

    size_mb = os.path.getsize(dest) / 1_000_000
    print(f"  [done]  {filename}  ({size_mb:.1f} MB)")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Saving data to: {DATA_DIR}\n")

    for cat in CATEGORIES:
        download_category(cat)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
