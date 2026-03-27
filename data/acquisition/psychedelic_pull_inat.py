#!/usr/bin/env python3
"""
Download psychedelics expert training images from iNaturalist API v1.

All taxon IDs verified against iNaturalist on 2026-03-27.

Targets are capped at ~80% of total observation count for low-count species
to leave room for research-grade filtering.

Run from repo root:
    python data/acquisition/psychedelic_pull_inat.py
"""
import requests
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("psychedelics_dataset")
BASE_DIR.mkdir(exist_ok=True)

# { (taxon_id, ...): ("folder_name", target_count) }
# All IDs verified 2026-03-27.
TAXA = {
    # ── Psilocybe species ────────────────────────────────────────────────────
    (328244,): ("psilocybe_cubensis",          3000),  # 4,662 obs
    (179085,): ("psilocybe_cyanescens",        2500),  # 3,510 obs
    (54025,):  ("psilocybe_semilanceata",      1500),  # 2,140 obs
    (348835,): ("psilocybe_ovoideocystidiata", 1800),  # 2,384 obs
    (206899,): ("psilocybe_caerulipes",         600),  # 866 obs  — capped
    (179652,): ("psilocybe_azurescens",         400),  # 568 obs  — capped

    # ── Other psychoactive / easily confused ─────────────────────────────────
    (83196,):  ("gymnopilus_junonius",         2000),  # 14,508 obs
    (418443,): ("panaeolus_cinctulus",         2500),  # 3,606 obs

    # ── Deadly lookalikes (oversampled for safety) ───────────────────────────
    (154735,): ("galerina_marginata_toxic",    5000),  # 24,269 obs
    (48715,):  ("amanita_muscaria_toxic",      3000),  # 147,058 obs
    (52135,):  ("amanita_phalloides_deadly",   5000),  # 12,880 obs
    # Pholiotina rugosa used as proxy — filaris (ID 1665287) has only 324 obs
    (877494,): ("conocybe_filaris_deadly",     2500),  # Pholiotina rugosa, 3,478 obs

    # ── Conservation ─────────────────────────────────────────────────────────
    (116333,): ("panax_quinquefolius_ginseng_conservation", 2500),  # 3,787 obs

    # ── OOD negative class ───────────────────────────────────────────────────
    (47170,):  ("other_mushroom",              3000),  # Fungi broad sample
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
    print(f"Downloading psychedelics dataset → {BASE_DIR}/")
    for taxon_ids, (folder, target) in TAXA.items():
        download_species(taxon_ids, folder, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
