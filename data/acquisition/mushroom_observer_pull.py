#!/usr/bin/env python3
"""
Download supplemental training images from Mushroom Observer API v2.

Used for species where iNaturalist has hit its observation ceiling.
MO attracts serious mycologists and often has distinct images not on iNat.

Currently targets:
  - psilocybe_azurescens  (~600 MO images)
  - psilocybe_caerulipes  (~600 MO images)

Images are saved into the same psychedelics_dataset/ folders as iNat images,
prefixed with "mo_" to avoid filename collisions.

Run from repo root:
    python data/acquisition/mushroom_observer_pull.py

MO API v2 docs: https://mushroomobserver.org/api2
"""

import time
from pathlib import Path
import requests
from tqdm import tqdm

BASE_URL = "https://mushroomobserver.org/api2"
DATASET_DIR = Path("psychedelics_dataset")
HEADERS = {"User-Agent": "ForagerMLBot/2.0 (homesteaderlabs_research)"}

# Min confidence for observation inclusion.
# MO confidence: -3 (rejected) to +3 (certain community ID).
# 1.5+ = solid consensus, 0.5+ = probable.
MIN_CONFIDENCE = 0.5

# Skip "All Rights Reserved" images — CC and public domain are fine for training.
BLOCKED_LICENSE_WORDS = {"rights reserved"}

# { "species_name_on_MO": ("dataset_folder", target_additional_images) }
TAXA = {
    "Psilocybe azurescens": ("psilocybe_azurescens", 500),
    "Psilocybe caerulipes": ("psilocybe_caerulipes", 500),
}


def _license_ok(license_str: str | None) -> bool:
    if not license_str:
        return True   # assume ok if unspecified
    return not any(w in license_str.lower() for w in BLOCKED_LICENSE_WORDS)


def _image_url(image_id: int) -> str:
    """640px CDN path — good quality without the overhead of originals."""
    return f"https://mushroomobserver.org/images/640/{image_id}.jpg"


def collect_image_ids(species_name: str) -> list[int]:
    """
    Page through all observations for species_name and collect image IDs.
    Filters by confidence and license.
    """
    image_ids: list[int] = []
    page = 1

    while True:
        try:
            r = requests.get(
                f"{BASE_URL}/observations",
                params={
                    "name": species_name,
                    "has_images": "true",
                    "format": "json",
                    "detail": "high",
                    "page": page,
                },
                headers=HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  API error (page {page}): {e}")
            time.sleep(5)
            continue

        if data.get("errors"):
            print(f"  API errors: {data['errors']}")
            break

        results = data.get("results") or []
        num_pages = data.get("number_of_pages", 1)

        for obs in results:
            if obs.get("confidence", 0) < MIN_CONFIDENCE:
                continue
            for img in obs.get("images") or []:
                if not _license_ok(img.get("license")):
                    continue
                image_ids.append(img["id"])

        if page >= num_pages:
            break
        page += 1
        time.sleep(0.4)

    return image_ids


def download_species(species_name: str, folder: str, target: int):
    save_path = DATASET_DIR / folder
    save_path.mkdir(parents=True, exist_ok=True)

    # Count existing MO images (prefixed mo_) to skip already-downloaded ones
    existing_mo = {f.stem for f in save_path.glob("mo_*.jpg")}
    already = len(existing_mo)

    if already >= target:
        print(f"  Skipping {folder} (MO) — already at {already}/{target}")
        return

    print(f"  Collecting image IDs for {species_name}...")
    image_ids = collect_image_ids(species_name)
    # Filter out already-downloaded
    image_ids = [i for i in image_ids if f"mo_{i}" not in existing_mo]
    print(f"  Found {len(image_ids)} new images (need {target - already} more)")

    needed = target - already
    pbar = tqdm(total=needed, desc=f"  {folder} (MO)", initial=0)
    downloaded = 0

    for img_id in image_ids:
        if downloaded >= needed:
            break
        url = _image_url(img_id)
        filepath = save_path / f"mo_{img_id}.jpg"
        if filepath.exists():
            continue
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            if resp.status_code == 200 and len(resp.content) > 5000:
                filepath.write_bytes(resp.content)
                downloaded += 1
                pbar.update(1)
            time.sleep(0.2)
        except Exception:
            continue

    pbar.close()
    total_inat = len(list(save_path.glob("*.jpg"))) - len(list(save_path.glob("mo_*.jpg")))
    total_mo = len(list(save_path.glob("mo_*.jpg")))
    print(f"  {folder}: {total_inat} iNat + {total_mo} MO = {total_inat + total_mo} total")


def main():
    print(f"Downloading Mushroom Observer supplemental images → {DATASET_DIR}/")
    print(f"Min confidence filter: {MIN_CONFIDENCE}")
    print()
    for species_name, (folder, target) in TAXA.items():
        download_species(species_name, folder, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
