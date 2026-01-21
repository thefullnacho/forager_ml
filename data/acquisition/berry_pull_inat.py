#!/usr/bin/env python3
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("berry_dataset")
BASE_DIR.mkdir(exist_ok=True)

# UPDATED TAXA: blueberry_wild now contains both Highbush (52733) and Lowbush (84542)
# Structure: { (tuple_of_ids): ("folder_name", target_count) }
TAXA = {
    (52733, 84542): ("blueberry_wild", 5000), 
    (47543,): ("blackberry_common", 5000), 
    (52684,): ("elderberry_american", 3000),
    (167829,): ("staghorn_sumac", 2000),
    (54498,): ("wild_grape_riverbank", 3000),
    (48599,): ("pokeweed_toxic", 4000),
    (50278,): ("virginia_creeper_toxic", 3000),
    (55620,): ("bittersweet_nightshade_toxic", 3000),
    (130900,): ("canada_moonseed_deadly", 5000),
    (58732,): ("poison_ivy", 5000)
}

headers = {"User-Agent": "SentinelForagerBot/1.0 (alex_homestead_project)"}

def download_images():
    for taxon_tuple, (folder, target) in TAXA.items():
        save_path = BASE_DIR / folder
        save_path.mkdir(exist_ok=True)
        
        # Check current count to skip full folders
        downloaded = len([f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))])
        
        if downloaded >= target:
            print(f"Skipping {folder} - already reached target ({downloaded}/{target})")
            continue
            
        taxon_ids_str = ",".join(map(str, taxon_tuple))
        page = 1
        pbar = tqdm(total=target, desc=f"Pulling {folder}", initial=downloaded)
        
        while downloaded < target:
            params = {
                "taxon_ids": taxon_ids_str,
                "quality_grade": "needs_id,research",
                "has[]": "photos",
                "per_page": 200,
                "page": page,
                "only_id": "false",
                "verifiable": "true",
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
                        if downloaded >= target: break
                        
                        img_url = photo['url'].replace("square", "medium")
                        img_filename = f"{photo['id']}.jpg"
                        
                        if not (save_path / img_filename).exists():
                            try:
                                img_data = requests.get(img_url, timeout=10).content
                                with open(save_path / img_filename, 'wb') as f:
                                    f.write(img_data)
                                downloaded += 1
                                pbar.update(1)
                            except:
                                continue
                
                page += 1
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"\nError on {folder}: {e}")
                time.sleep(5)
                continue

if __name__ == "__main__":
    download_images()
