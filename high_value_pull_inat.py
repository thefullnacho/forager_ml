#!/usr/bin/env python3
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("high_value_dataset")
BASE_DIR.mkdir(exist_ok=True)

# UPDATED TAXA - Unlocked for maximum hits
TAXA = {
    (48398, 48399): ("ginseng_american", 5000),         
    (56830,): ("morels_edible", 5000),             
    (54271, 1066708, 158229): ("ramps_wild_leek", 4000),           
    (47347,): ("chanterelles_edible", 4000),       
    (82528,): ("saffron_crocus", 2000),            
    (49158,): ("lions_mane", 3000),                
    (48431,): ("chicken_of_the_woods", 3000),      
    (127051,): ("chaga_medicinal", 3000),          
    (47743,): ("ostrich_fern_fiddlehead", 3000),   
    (63274, 350824): ("reishi_northeast", 3000),   # Combined Hemlock & Hardwood Reishi
    (85468, 48532, 129707, 204561): ("high_value_toxics", 5000) 
}

headers = {"User-Agent": "SentinelForagerBot/1.0 (high_value_harvest)"}

def download_images():
    for taxon_tuple, (folder, target) in TAXA.items():
        save_path = BASE_DIR / folder
        save_path.mkdir(exist_ok=True)
        
        # SKIP LOGIC: Don't redo work
        downloaded = len(list(save_path.glob("*.jpg")))
        if downloaded >= target:
            print(f"Skipping {folder} - already full ({downloaded}/{target})")
            continue
            
        taxon_ids_str = ",".join(map(str, taxon_tuple))
        page = 1
        pbar = tqdm(total=target, desc=f"Pulling {folder}", initial=downloaded)
        
        while downloaded < target:
            params = {
                "taxon_ids": taxon_ids_str,
                "quality_grade": "research,needs_id", # UNLOCKED
                "has[]": "photos",
                "verifiable": "any",
                "per_page": 200,
                "page": page,
                "order_by": "created_at" 
            }
            
            try:
                r = requests.get("https://api.inaturalist.org/v1/observations", params=params, headers=headers, timeout=15)
                data = r.json()
                results = data.get("results", [])
                if not results: break
                
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
                            except: continue
                page += 1
                time.sleep(0.3) 
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(5)
                continue

if __name__ == "__main__":
    download_images()
