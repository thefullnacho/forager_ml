#!/usr/bin/env python3
"""
Download "other" class images for the domain router from iNaturalist API v1.

The "other" class is the router's first-pass OOD gate — it catches anything
that isn't a berry, mushroom, or plant so the expensive expert models never
run on non-target inputs.

Targets a diverse set of non-foraging subjects a field user might encounter:
  - Birds, mammals, reptiles, amphibians, insects, arachnids
  - Lichens (plant-like but not medicinal/edible target)
  - Aquatic/marine life
  - Soil/rock/geology observations

Images are saved to inat_dataset/other/<subclass>/ and aggregated by
build_router_dataset.py when rebuilding the router training split.

Run from repo root:
    python data/acquisition/other_pull_inat.py
"""

import requests
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("inat_dataset") / "other"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# { taxon_id: ("folder_name", target_count) }
# All IDs verified against iNaturalist. taxon IDs are broad groups — we want
# diversity within each group so we pull with order_by=random (via created_at
# with high page count to effectively randomize).
TAXA = {
    # ── Animals (encountered in the field) ────────────────────────────────
    3:      ("birds",        3000),   # Aves — very common field encounter
    40151:  ("mammals",      2500),   # Mammalia — deer, squirrels, etc.
    26036:  ("reptiles",     1500),   # Reptilia
    20978:  ("amphibians",   1000),   # Amphibia
    # ── Arthropods ───────────────────────────────────────────────────────
    47158:  ("insects",      3000),   # Insecta — hard negative for plants
    47119:  ("arachnids",    1000),   # Arachnida (spiders, ticks)
    # ── Non-target botanicals ─────────────────────────────────────────────
    54743:  ("lichens",      2000),   # Lecanoromycetes — plant-like, challenging
    311313: ("mosses",       1000),   # Bryophyta — moss, plant-like texture
    # ── Aquatic ───────────────────────────────────────────────────────────
    47178:  ("fish",         1500),   # Actinopterygii
    47549:  ("marine_invert", 1000),  # Marine invertebrates
    # ── Inanimate / challenging backgrounds ─────────────────────────────
    # iNat doesn't have rocks/soil well-labelled, but fungi textures can be
    # confusing; cup fungi and puffballs are legitimate "hard other" targets
    52750:  ("cup_fungi",    1000),   # Pezizomycetes — cup fungi look plant-like
    # Slime molds — common misidentification target
    47682:  ("slime_molds",   500),   # Myxogastria
}

headers = {"User-Agent": "ForagerMLBot/2.0 (homesteaderlabs_research)"}


def download_taxon(taxon_id: int, folder: str, target: int):
    save_path = BASE_DIR / folder
    save_path.mkdir(exist_ok=True)

    downloaded = len(list(save_path.glob("*.jpg")))
    if downloaded >= target:
        print(f"  Skipping {folder} — already at {downloaded}/{target}")
        return

    page = 1
    pbar = tqdm(total=target, desc=f"  {folder}", initial=downloaded)

    while downloaded < target:
        params = {
            "taxon_id": taxon_id,
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
                            if len(img_data) > 5000:
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
    total_target = sum(t for _, (_, t) in TAXA.items())
    print(f"Downloading 'other' class images → {BASE_DIR}/")
    print(f"{len(TAXA)} subclasses — target {total_target:,} total images\n")
    for taxon_id, (folder, target) in TAXA.items():
        download_taxon(taxon_id, folder, target)
    print("\nDone.")
    print(f"\nNext: run python training/scripts/build_router_dataset.py to rebuild router_dataset/")


if __name__ == "__main__":
    main()
