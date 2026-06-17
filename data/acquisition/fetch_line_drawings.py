"""
fetch_line_drawings.py — Download and process species line drawings for the
Forager e-ink display.

Prioritizes botanical illustrations and line drawings from Wikimedia Commons.
Outputs 1-bit (B&W) PNGs with transparent backgrounds.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

try:
    import requests
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    print("Missing dependencies. Run: pip install requests Pillow")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "inference" / "illustrations"

# Target size
ILLUS_W = 184
ILLUS_H = 210

HEADERS = {
    "User-Agent": "ForagerML/1.0 (botanical field identification device; educational use)"
}

# Species list
SPECIES = {
    "Rubus allegheniensis": "blackberry_common",
    "Vaccinium corymbosum": "blueberry_highbush",
    "Vaccinium angustifolium": "blueberry_wild",
    "Sambucus canadensis": "elderberry_american",
    "Rhus typhina": "staghorn_sumac",
    "Vitis riparia": "wild_grape_riverbank",
    "Solanum dulcamara": "bittersweet_nightshade_toxic",
    "Menispermum canadense": "canada_moonseed_deadly",
    "Toxicodendron radicans": "poison_ivy",
    "Phytolacca americana": "pokeweed_toxic",
    "Parthenocissus quinquefolia": "virginia_creeper_toxic",
    "Cantharellus cibarius": "chanterelles_edible",
    "Morchella esculenta": "morels_edible",
    "Laetiporus sulphureus": "chicken_of_the_woods",
    "Hericium erinaceus": "lions_mane",
    "Inonotus obliquus": "chaga_medicinal",
    "Ganoderma lucidum": "reishi_mushroom",
    "Ganoderma tsugae": "reishi_northeast",
    "Allium tricoccum": "ramps_wild_leek",
    "Matteuccia struthiopteris": "ostrich_fern_fiddlehead",
    "Panax quinquefolius": "ginseng_american",
    "Crocus sativus": "saffron_crocus",
    "Amanita phalloides": "amanita_phalloides_deadly",
    "Amanita muscaria": "amanita_muscaria_toxic",
    "Galerina marginata": "galerina_marginata_toxic",
    "Conocybe filaris": "conocybe_filaris_deadly",
    "Gymnopilus junonius": "gymnopilus_junonius",
    "Panaeolus cinctulus": "panaeolus_cinctulus",
    "Psilocybe ovoideocystidiata": "psilocybe_ovoideocystidiata",
    "Psilocybe cubensis": "psilocybe_cubensis",
    "Psilocybe cyanescens": "psilocybe_cyanescens",
    "Psilocybe semilanceata": "psilocybe_semilanceata",
    "Psilocybe azurescens": "psilocybe_azurescens",
    "Psilocybe caerulipes": "psilocybe_caerulipes",
    "Eupatorium perfoliatum": "boneset",
    "Arctium lappa": "burdock",
    "Nepeta cataria": "catnip",
    "Tussilago farfara": "coltsfoot",
    "Echinacea purpurea": "echinacea",
    "Digitalis purpurea": "foxglove_toxic",
    "Solidago canadensis": "goldenrod",
    "Leonurus cardiaca": "motherwort",
    "Verbascum thapsus": "mullein",
    "Plantago major": "plantain_broadleaf",
    "Conium maculatum": "poison_hemlock_deadly",
    "Trifolium pratense": "red_clover",
    "Hypericum perforatum": "st_johns_wort",
    "Urtica dioica": "stinging_nettle",
    "Valeriana officinalis": "valerian",
    "Cicuta maculata": "water_hemlock_deadly",
    "Ageratina altissima": "white_snakeroot_toxic",
    "Monarda fistulosa": "wild_bergamot",
    "Daucus carota": "wild_carrot",
    "Laportea canadensis": "wood_nettle",
    "Achillea millefolium": "yarrow",
}

def get_commons_image_url(file_title: str) -> str | None:
    url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles={quote(file_title)}"
        f"&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
    )
    data = fetch_api(url)
    if not data:
        return None
        
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])
        if info and info[0].get("thumburl"):
            return info[0]["thumburl"]
    return None

def fetch_api(url: str, retries: int = 5) -> dict | None:
    """Fetch from MediaWiki API with exponential backoff for 429s."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    Rate limited (429) — waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print(f"    API error: {e}")
    return None

def fetch_commons_illustration_url(scientific_name: str) -> str | None:
    # Try more specific terms to find illustrations/drawings
    for term in ["illustration", "drawing", "plate", "sketch"]:
        search_term = f"{scientific_name} {term}"
        
        url = (
            f"https://commons.wikimedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote(search_term)}"
            f"&srnamespace=6&srlimit=10&format=json"
        )
        
        data = fetch_api(url)
        if not data:
            continue
            
        results = data.get("query", {}).get("search", [])
        illustration_keywords = {"illustration", "drawing", "plate", "kohler", "thome", "flora", "sketch", "line", "botanical"}
        blacklist = {"design drawing", "pattern", "ornament", "architecture"}

        for result in results:
            title = result["title"].lower()
            
            # Must contain an illustration keyword
            if not any(kw in title for kw in illustration_keywords):
                continue
                
            # Must NOT contain blacklist keywords
            if any(bl in title for bl in blacklist):
                continue
                
            print(f"    Found on Commons: {result['title']}")
            img_url = get_commons_image_url(result["title"])
            if img_url:
                return img_url
            
    return None

def fetch_wikipedia_image_url(scientific_name: str) -> str | None:
    title = scientific_name.replace(" ", "_")
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={quote(title)}"
        f"&prop=pageimages&pithumbsize=800&format=json&redirects=1"
    )
    data = fetch_api(url)
    if not data:
        return None
        
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        if thumb.get("source"):
            return thumb["source"]
    return None

def download_image(url: str, retries: int = 3) -> Image.Image | None:
    from io import BytesIO
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).copy()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print(f"    Download error: {e}")
    return None

def process_line_drawing(img: Image.Image) -> Image.Image:
    """
    Convert image to 1-bit black and white with transparency.
    """
    # 1. Convert to grayscale
    img = img.convert("L")
    
    # 2. Auto-contrast and sharpen to emphasize lines
    img = ImageOps.autocontrast(img, cutoff=5)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    
    # 3. Resize with aspect ratio preserved
    img.thumbnail((ILLUS_W, ILLUS_H), Image.LANCZOS)
    
    # 4. Threshold to 1-bit (B&W)
    # We want lines to be black, background to be white (which we'll make transparent)
    threshold = 180 # Adjust threshold as needed
    img = img.point(lambda p: 255 if p > threshold else 0)
    
    # 5. Create transparent background
    # Create RGBA image with WHITE background but 0 alpha
    # This ensures that img.convert("L") will treat the background as white (255)
    rgba = Image.new("RGBA", (ILLUS_W, ILLUS_H), (255, 255, 255, 0))
    
    # Center the image on the canvas
    x_off = (ILLUS_W - img.width) // 2
    y_off = (ILLUS_H - img.height) // 2
    
    # Paste the black lines into the RGBA image
    # We use the 1-bit image as a mask: black pixels in 'img' (lines) become solid black in 'rgba'
    # White pixels in 'img' (background) remain transparent in 'rgba'
    line_color = (0, 0, 0, 255)
    mask = ImageOps.invert(img) # Lines are now 255, background is 0
    
    lines = Image.new("RGBA", (img.width, img.height), line_color)
    rgba.paste(lines, (x_off, y_off), mask)
    
    return rgba

def fetch_one(scientific_name: str, species_key: str, dry_run: bool, force: bool) -> bool:
    out_path = OUTPUT_DIR / f"{species_key}.png"

    if out_path.exists() and not force:
        print(f"  ✓ {species_key} — exists, skipping")
        return True

    print(f"  {species_key} ({scientific_name})")

    if dry_run:
        return True

    # 1. Prefer Commons illustration/drawing search
    img_url = fetch_commons_illustration_url(scientific_name)
    source = "Commons (Illustration)"

    # 2. Fall back to Wikipedia page image (less likely to be a line drawing)
    if not img_url:
        print(f"    No illustration found on Commons — trying Wikipedia...")
        img_url = fetch_wikipedia_image_url(scientific_name)
        source = "Wikipedia"

    if not img_url:
        print(f"    ✗ No image found for {scientific_name}")
        return False

    print(f"    Downloading from {source}: {img_url[:60]}...")
    raw = download_image(img_url)
    if raw is None:
        return False

    processed = process_line_drawing(raw)
    processed.save(out_path, "PNG")
    print(f"    ✓ Saved: {out_path.name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch species line drawings")
    parser.add_argument("--only", help="Fetch only this species key")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--force", action="store_true", help="Force refetch")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = SPECIES
    if args.only:
        targets = {sci: key for sci, key in SPECIES.items() if key == args.only}

    success, failed = 0, []
    for sci, key in targets.items():
        if fetch_one(sci, key, args.dry_run, args.force):
            success += 1
        else:
            failed.append(key)
        if not args.dry_run:
            time.sleep(5.0) # Increased delay to 5s

    print(f"\nDone. {success}/{len(targets)} fetched.")
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
