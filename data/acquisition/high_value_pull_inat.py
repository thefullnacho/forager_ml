#!/usr/bin/env python3
"""
Download high-value expert training images from iNaturalist API v1.

All taxon IDs verified against iNaturalist on 2026-03-27.

Run from repo root:
    python data/acquisition/high_value_pull_inat.py
"""
import requests
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("high_value_dataset")
BASE_DIR.mkdir(exist_ok=True)

# { (taxon_id, ...): ("folder_name", target_count) }
# All IDs verified 2026-03-27.
TAXA = {
    # ── High-value edibles ───────────────────────────────────────────────────
    (47347,):          ("chanterelles_edible",      4000),  # Cantharellus, 47,347 obs
    (56830,):          ("morels_edible",            4000),  # Morchella, 5,000+ obs
    (49158,):          ("lions_mane",               3000),  # Hericium erinaceus, 14,000+ obs
    (48431,):          ("chicken_of_the_woods",     3000),  # Laetiporus, 30,000+ obs
    (127021,):         ("chaga_medicinal",          3000),  # Inonotus obliquus, 9,613 obs
    # Reishi: Ganoderma tsugae (63274, hemlock) + G. sessile (350824, hardwood)
    (63274, 350824):   ("reishi_northeast",         3000),  # 21,549+ obs
    (116333,):         ("ginseng_american",         2500),  # Panax quinquefolius, 3,787 obs
    # Ramps: Allium tricoccum (55634) + A. tricoccum var. burdickii (white ramps)
    (55634,):          ("ramps_wild_leek",          3000),  # 18,480 obs
    (82574,):          ("ostrich_fern_fiddlehead",  3000),  # Matteuccia struthiopteris, 42,961 obs
    # Saffron crocus: only 361 research-grade obs — use needs_id too, cap low
    (118933,):         ("saffron_crocus",            250),  # Crocus sativus, 361 obs — capped

    # ── Toxic lookalikes (oversampled for safety) ────────────────────────────
    # Composite class: false morel + death cap + destroying angels + funeral bell + toxic lepiota
    (85120, 52135, 67356, 125390, 154735, 58694): ("high_value_toxics", 5000),
}

headers = {"User-Agent": "ForagerMLBot/2.0 (homesteaderlabs_research)"}


def download_species(taxon_ids: tuple, folder: str, target: int, quality: str = "research"):
    save_path = BASE_DIR / folder
    save_path.mkdir(exist_ok=True)

    downloaded = len(list(save_path.glob("*.jpg")))
    if downloaded >= target:
        print(f"  Skipping {folder} — already at {downloaded}/{target}")
        return

    taxon_ids_str = ",".join(map(str, taxon_ids))
    page = 1
    pbar = tqdm(total=target, desc=f"  {folder}", initial=downloaded)

    while downloaded < target:
        params = {
            "taxon_id": taxon_ids_str,
            "quality_grade": quality,
            "has[]": "photos",
            "verifiable": "true",
            "per_page": 200,
            "page": page,
            "order_by": "created_at",
        }
        try:
            r = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params=params, headers=headers, timeout=15
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                break

            for obs in results:
                for photo in obs.get("photos", []):
                    if downloaded >= target:
                        break
                    img_url = photo["url"].replace("square", "medium")
                    img_filename = f"{photo['id']}.jpg"
                    filepath = save_path / img_filename
                    if not filepath.exists():
                        try:
                            img_data = requests.get(img_url, timeout=10).content
                            with open(filepath, "wb") as f:
                                f.write(img_data)
                            downloaded += 1
                            pbar.update(1)
                        except Exception:
                            continue

            page += 1
            time.sleep(0.4)

        except Exception as e:
            print(f"\n  API error ({folder}): {e}")
            time.sleep(5)

    pbar.close()
    print(f"  {folder}: {downloaded} images")


def main():
    print(f"Downloading high-value dataset → {BASE_DIR}/")
    for taxon_ids, (folder, target) in TAXA.items():
        # saffron_crocus is so rare we relax to needs_id too
        quality = "research,needs_id" if folder == "saffron_crocus" else "research"
        download_species(taxon_ids, folder, target, quality=quality)
    print("\nDone.")


if __name__ == "__main__":
    main()
