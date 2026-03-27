#!/usr/bin/env python3
"""
Download berry expert training images from iNaturalist API v1.

All taxon IDs verified against iNaturalist on 2026-03-27.

Run from repo root:
    python data/acquisition/berry_pull_inat.py
"""
import requests
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("berry_dataset")
BASE_DIR.mkdir(exist_ok=True)

# { (taxon_id, ...): ("folder_name", target_count) }
# All IDs verified 2026-03-27.
TAXA = {
    # ── Edible berries ───────────────────────────────────────────────────────
    # Wild blueberry: Vaccinium angustifolium (84542) + V. myrtilloides (60942)
    (84542, 60942):    ("blueberry_wild",               4000),  # 27,681 + obs
    (52740,):          ("blueberry_highbush",            4000),  # 21,244 obs
    # Blackberry: Rubus allegheniensis (82110) + R. occidentalis black raspberry (82111)
    (82110, 82111):    ("blackberry_common",             4000),  # 33,156+ obs
    # Elderberry: Sambucus canadensis (84300) + S. nigra (765394)
    (84300, 765394):   ("elderberry_american",           4000),  # 68,407+ obs
    (167829,):         ("staghorn_sumac",                3000),  # 79,719 obs
    (119936,):         ("wild_grape_riverbank",          3000),  # 39,465 obs

    # ── Toxic lookalikes (oversampled for safety) ────────────────────────────
    (48599,):          ("pokeweed_toxic",                4000),  # 160,899 obs
    (50278,):          ("virginia_creeper_toxic",        3000),  # 140,516 obs
    (55620,):          ("bittersweet_nightshade_toxic",  3000),  # 113,908 obs
    (130900,):         ("canada_moonseed_deadly",        3000),  # 14,090 obs  ✓ correct here
    (58732,):          ("poison_ivy",                    4000),  # 146,801 obs
}

headers = {"User-Agent": "ForagerMLBot/2.0 (homesteaderlabs_research)"}


def download_species(taxon_ids: tuple, folder: str, target: int):
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
            "quality_grade": "research",
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
    print(f"Downloading berry dataset → {BASE_DIR}/")
    for taxon_ids, (folder, target) in TAXA.items():
        download_species(taxon_ids, folder, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
