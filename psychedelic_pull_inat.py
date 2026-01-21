#!/usr/bin/env python3
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("psychedelics_dataset")
BASE_DIR.mkdir(exist_ok=True)

# BROADENED TAXA - genus-level for low-count species
TAXA = {
    # Core Northeast psychedelics
    130902: ("psilocybe_caerulipes", 3000),
    130903: ("psilocybe_ovoideocystidiata", 1000),
    130898: ("psilocybe_genus", 5000),              # Broad boost for ovoideocystidiata & others
    130901: ("psilocybe_semilanceata", 3000),
    47545: ("gymnopilus_junonius", 2000),
    130904: ("panaeolus_cinctulus", 1000),
    130899: ("panaeolus_genus", 5000),              # Broad boost for cinctulus
    130899: ("psilocybe_cubensis", 4000),           # Global classic
    130900: ("psilocybe_cyanescens", 3000),
    130898: ("psilocybe_azurescens", 2000),

    # Toxics & deadly lookalikes (oversampled)
    121487: ("galerina_marginata_toxic", 5000),
    48460: ("amanita_muscaria_toxic", 3000),
    126699: ("amanita_phalloides_deadly", 5000),
    130906: ("conocybe_filaris_deadly", 2000),

    # Conservation & negatives
    130905: ("panax_quinquefolius_ginseng_conservation", 3000),
    47126: ("other_mushroom", 2000),                # Broad non-psyche fungi
}

headers = {"User-Agent": "SentinelForagerBot/1.0 (psychedelic_research)"}

def download_images():
    for taxon_id, (folder, target) in TAXA.items():
        save_path = BASE_DIR / folder
        save_path.mkdir(exist_ok=True)
        
        # Skip if already full
        downloaded = len(list(save_path.glob("*.jpg")))
        if downloaded >= target:
            print(f"Skipping {folder} - already full ({downloaded}/{target})")
            continue
            
        taxon_ids_str = str(taxon_id)  # Single ID
        
        page = 1
        pbar = tqdm(total=target, desc=f"Pulling {folder}", initial=downloaded)
        
        while downloaded < target:
            params = {
                "taxon_ids": taxon_ids_str,
                "quality_grade": "needs_id,research",  # ← Changed back as requested
                "has[]": "photos",
                "verifiable": "true",
                "per_page": 200,
                "page": page,
                "order_by": "created_at"
            }
            
            try:
                r = requests.get("https://api.inaturalist.org/v1/observations", params=params, headers=headers, timeout=15)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                
                for obs in results:
                    for photo in obs.get("photos", []):
                        if downloaded >= target:
                            break
                        img_url = photo['url'].replace("square", "medium")
                        img_filename = f"{photo['id']}.jpg"
                        
                        filepath = save_path / img_filename
                        if not filepath.exists():
                            try:
                                img_data = requests.get(img_url, timeout=10).content
                                with open(filepath, 'wb') as f:
                                    f.write(img_data)
                                downloaded += 1
                                pbar.update(1)
                            except Exception as e:
                                print(f"Download failed for {img_filename}: {e}")
                page += 1
                time.sleep(0.3)  # Rate limit safety
            except Exception as e:
                print(f"\nAPI error: {e}")
                time.sleep(5)
                continue

    print("\nPull complete! Check folder counts.")

if __name__ == "__main__":
    download_images()
