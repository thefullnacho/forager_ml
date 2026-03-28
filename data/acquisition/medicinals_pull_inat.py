#!/usr/bin/env python3
"""
Download medicinals expert training images from iNaturalist API v1.

All taxon IDs verified against iNaturalist on 2026-03-28.

21 classes: 17 safe/medicinal species + 4 deadly/toxic lookalikes.

Key dangerous lookalike pairs in this dataset:
  yarrow / wild_carrot  →  poison_hemlock_deadly, water_hemlock_deadly
  boneset               →  white_snakeroot_toxic
  mullein (rosette)     →  foxglove_toxic

Run from repo root:
    python data/acquisition/medicinals_pull_inat.py
"""
import requests
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("medicinals_dataset")
BASE_DIR.mkdir(exist_ok=True)

# { (taxon_id, ...): ("folder_name", target_count) }
# All IDs verified 2026-03-28.
TAXA = {
    # ── Safe medicinals ───────────────────────────────────────────────────────
    (51884,):  ("stinging_nettle",        4000),  # 133,178 obs — Urtica dioica
    (119802,): ("wood_nettle",            3000),  # 25,067 obs  — Laportea canadensis
    (52821,):  ("yarrow",                 4000),  # 293,946 obs — Achillea millefolium
    (59029,):  ("mullein",                4000),  # 162,377 obs — Verbascum thapsus
    (67808,):  ("goldenrod",              4000),  # 60,029 obs  — Solidago canadensis
    (56077,):  ("st_johns_wort",          4000),  # 114,024 obs — Hypericum perforatum
    (58961,):  ("plantain_broadleaf",     4000),  # 112,674 obs — Plantago major
    (59570,):  ("burdock",                3000),  # 43,141 obs  — Arctium minus
    (119045,): ("boneset",                3000),  # 33,019 obs  — Eupatorium perfoliatum
    (51875,):  ("red_clover",             4000),  # 207,940 obs — Trifolium pratense
    (85320,):  ("wild_bergamot",          3000),  # 71,062 obs  — Monarda fistulosa
    (48622,):  ("catnip",                 3000),  # 23,736 obs  — Nepeta cataria
    (56171,):  ("motherwort",             3000),  # 35,384 obs  — Leonurus cardiaca
    (56160,):  ("valerian",               3000),  # 33,279 obs  — Valeriana officinalis
    (48627,):  ("echinacea",              3000),  # 45,693 obs  — Echinacea purpurea
    (56222,):  ("coltsfoot",              4000),  # 119,655 obs — Tussilago farfara
    (76610,):  ("wild_carrot",            4000),  # 183,795 obs — Daucus carota

    # ── Deadly / toxic lookalikes (oversampled for safety) ────────────────────
    # Primary confusion: yarrow + wild_carrot → hemlock family
    (52998,):  ("poison_hemlock_deadly",  5000),  # 57,962 obs  — Conium maculatum
    (60125,):  ("water_hemlock_deadly",   3000),  # 16,984 obs  — Cicuta maculata
    # boneset → white snakeroot (caused milk sickness)
    (119048,): ("white_snakeroot_toxic",  4000),  # 103,391 obs — Ageratina altissima
    # mullein rosette → foxglove rosette
    (53983,):  ("foxglove_toxic",         4000),  # 93,521 obs  — Digitalis purpurea
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
    total_target = sum(t for _, (_, t) in TAXA.items())
    print(f"Downloading medicinals dataset → {BASE_DIR}/")
    print(f"21 classes — target {total_target:,} total images\n")
    for taxon_ids, (folder, target) in TAXA.items():
        download_species(taxon_ids, folder, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
