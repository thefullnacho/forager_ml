"""
fetch_illustrations.py — Download and process species illustrations for the
Forager e-ink display.

For each species in SPECIES_METADATA, fetches the Wikipedia page image by
scientific name, converts to 4-gray grayscale with Floyd-Steinberg dithering,
and saves to inference/illustrations/{species_key}.png at display resolution.

Sources (tried in order):
  1. Wikipedia page image (reliable, broad coverage)
  2. Wikipedia Commons search for "[scientific name] illustration" (better for
     historical botanical plates — often finds BHL/Wikisource drawings)

Output images are pre-processed for the Waveshare 3.7" e-ink display:
  - 184 x 210 px (fits the illustration zone in display.py)
  - 4 gray levels (0, 85, 170, 255) with Floyd-Steinberg dithering
  - 'L' mode PNG

Usage:
    python data/acquisition/fetch_illustrations.py

    # Single species (for testing):
    python data/acquisition/fetch_illustrations.py --only blackberry_common

    # Dry run (show what would be fetched, no downloads):
    python data/acquisition/fetch_illustrations.py --dry-run
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

# Target size for the illustration zone in display.py
ILLUS_W = 184
ILLUS_H = 210

# 4-gray palette values matching the e-ink display
GRAY_LEVELS = [0, 85, 170, 255]

HEADERS = {
    "User-Agent": "ForagerML/1.0 (botanical field identification device; educational use)"
}

# ── Species list from convergence.py ──────────────────────────────────────────
# scientific name → species key (used as filename)
SPECIES = {
    # Berry expert
    "Rubus allegheniensis":        "blackberry_common",
    "Vaccinium corymbosum":        "blueberry_highbush",
    "Vaccinium angustifolium":     "blueberry_wild",
    "Sambucus canadensis":         "elderberry_american",
    "Rhus typhina":                "staghorn_sumac",
    "Vitis riparia":               "wild_grape_riverbank",
    "Solanum dulcamara":           "bittersweet_nightshade_toxic",
    "Menispermum canadense":       "canada_moonseed_deadly",
    "Toxicodendron radicans":      "poison_ivy",
    "Phytolacca americana":        "pokeweed_toxic",
    "Parthenocissus quinquefolia": "virginia_creeper_toxic",

    # High-value expert
    "Cantharellus cibarius":       "chanterelles_edible",
    "Morchella esculenta":         "morels_edible",
    "Laetiporus sulphureus":       "chicken_of_the_woods",
    "Hericium erinaceus":          "lions_mane",
    "Inonotus obliquus":           "chaga_medicinal",
    "Ganoderma lucidum":           "reishi_mushroom",
    "Ganoderma tsugae":            "reishi_northeast",
    "Allium tricoccum":            "ramps_wild_leek",
    "Matteuccia struthiopteris":   "ostrich_fern_fiddlehead",
    "Panax quinquefolius":         "ginseng_american",
    "Crocus sativus":              "saffron_crocus",

    # Psychedelics expert
    "Amanita phalloides":          "amanita_phalloides_deadly",
    "Amanita muscaria":            "amanita_muscaria_toxic",
    "Galerina marginata":          "galerina_marginata_toxic",
    "Conocybe filaris":            "conocybe_filaris_deadly",
    "Gymnopilus junonius":         "gymnopilus_junonius",
    "Panaeolus cinctulus":         "panaeolus_cinctulus",
    "Psilocybe ovoideocystidiata": "psilocybe_ovoideocystidiata",
    "Psilocybe cubensis":          "psilocybe_cubensis",
    "Psilocybe cyanescens":        "psilocybe_cyanescens",
    "Psilocybe semilanceata":      "psilocybe_semilanceata",
    "Psilocybe azurescens":        "psilocybe_azurescens",
    "Psilocybe caerulipes":        "psilocybe_caerulipes",

    # Medicinals expert
    "Eupatorium perfoliatum":      "boneset",
    "Arctium lappa":               "burdock",
    "Nepeta cataria":              "catnip",
    "Tussilago farfara":           "coltsfoot",
    "Echinacea purpurea":          "echinacea",
    "Digitalis purpurea":          "foxglove_toxic",
    "Solidago canadensis":         "goldenrod",
    "Leonurus cardiaca":           "motherwort",
    "Verbascum thapsus":           "mullein",
    "Plantago major":              "plantain_broadleaf",
    "Conium maculatum":            "poison_hemlock_deadly",
    "Trifolium pratense":          "red_clover",
    "Hypericum perforatum":        "st_johns_wort",
    "Urtica dioica":               "stinging_nettle",
    "Valeriana officinalis":       "valerian",
    "Cicuta maculata":             "water_hemlock_deadly",
    "Ageratina altissima":         "white_snakeroot_toxic",
    "Monarda fistulosa":           "wild_bergamot",
    "Daucus carota":               "wild_carrot",
    "Laportea canadensis":         "wood_nettle",
    "Achillea millefolium":        "yarrow",
}


# ── Image fetching ─────────────────────────────────────────────────────────────

def fetch_wikipedia_image_url(scientific_name: str) -> str | None:
    """Get the main image URL from a Wikipedia article by scientific name."""
    title = scientific_name.replace(" ", "_")
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={quote(title)}"
        f"&prop=pageimages&pithumbsize=500&format=json&redirects=1"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {})
            if thumb.get("source"):
                return thumb["source"]
    except Exception as e:
        print(f"    Wikipedia API error: {e}")
    return None


def fetch_commons_illustration_url(scientific_name: str) -> str | None:
    """
    Search Wikimedia Commons for a botanical illustration of the species.
    Prefers results with 'illustration', 'drawing', or 'plate' in the filename
    since these tend to be historical botanical art that renders well on e-ink.
    """
    search_term = f"{scientific_name} botanical illustration"
    url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote(search_term)}"
        f"&srnamespace=6&srlimit=5&format=json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])

        # Prefer results that look like illustrations
        illustration_keywords = {"illustration", "drawing", "plate", "kohler", "thome", "flora"}
        for result in results:
            title = result["title"].lower()
            if any(kw in title for kw in illustration_keywords):
                # Get the actual image URL
                img_url = get_commons_image_url(result["title"])
                if img_url:
                    return img_url

        # Fall back to first result
        if results:
            return get_commons_image_url(results[0]["title"])
    except Exception as e:
        print(f"    Commons search error: {e}")
    return None


def get_commons_image_url(file_title: str) -> str | None:
    """Get the direct download URL for a Wikimedia Commons file."""
    url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles={quote(file_title)}"
        f"&prop=imageinfo&iiprop=url&iiurlwidth=500&format=json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])
            if info and info[0].get("thumburl"):
                return info[0]["thumburl"]
    except Exception:
        pass
    return None


def download_image(url: str, retries: int = 3) -> Image.Image | None:
    """Download an image from a URL and return as PIL Image. Retries on 429."""
    from io import BytesIO
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s ...")
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


# ── Image processing ───────────────────────────────────────────────────────────

def process_for_eink(img: Image.Image) -> Image.Image:
    """
    Convert an image to 4-gray e-ink format:
      1. Convert to grayscale
      2. Auto-contrast to spread tonal range
      3. Resize to ILLUS_W x ILLUS_H with aspect-preserving letterbox
      4. Floyd-Steinberg dither to 4 gray levels
    """
    # Grayscale + auto-contrast
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)

    # Boost contrast slightly — helps line drawings pop on e-ink
    img = ImageEnhance.Contrast(img).enhance(1.3)

    # Resize with aspect ratio preserved, letterboxed on white
    img.thumbnail((ILLUS_W, ILLUS_H), Image.LANCZOS)
    canvas = Image.new("L", (ILLUS_W, ILLUS_H), 255)
    x_off = (ILLUS_W - img.width) // 2
    y_off = (ILLUS_H - img.height) // 2
    canvas.paste(img, (x_off, y_off))

    # Dither to 4 gray levels using Floyd-Steinberg
    palette_img = Image.new("P", (1, 1))
    flat_palette = []
    for v in GRAY_LEVELS:
        flat_palette += [v, v, v]
    flat_palette += [0] * (768 - len(flat_palette))
    palette_img.putpalette(flat_palette)

    dithered = canvas.quantize(colors=4, palette=palette_img,
                               dither=Image.Dither.FLOYDSTEINBERG)
    return dithered.convert("L")


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_one(scientific_name: str, species_key: str, dry_run: bool) -> bool:
    """Fetch and process one species illustration. Returns True on success."""
    out_path = OUTPUT_DIR / f"{species_key}.png"

    if out_path.exists():
        print(f"  ✓ {species_key} — already exists, skipping")
        return True

    print(f"  {species_key} ({scientific_name})")

    if dry_run:
        print(f"    [dry run] would fetch from Wikipedia/Commons")
        return True

    # Try Wikipedia page image first
    img_url = fetch_wikipedia_image_url(scientific_name)
    source = "Wikipedia"

    # Fall back to Commons illustration search
    if not img_url:
        print(f"    Wikipedia: no image — trying Commons illustration search...")
        img_url = fetch_commons_illustration_url(scientific_name)
        source = "Commons"

    if not img_url:
        print(f"    ✗ No image found for {scientific_name}")
        return False

    print(f"    Downloading from {source}: {img_url[:80]}...")
    raw = download_image(img_url)
    if raw is None:
        return False

    processed = process_for_eink(raw)
    processed.save(out_path, "PNG")
    print(f"    ✓ Saved: {out_path.name}  ({processed.size[0]}×{processed.size[1]})")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Fetch species illustrations for e-ink display")
    p.add_argument("--only", help="Fetch only this species key (e.g. blackberry_common)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be fetched without downloading")
    p.add_argument("--refetch", action="store_true", help="Re-fetch even if file already exists")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = SPECIES
    if args.only:
        # Find by species key
        match = {sci: key for sci, key in SPECIES.items() if key == args.only}
        if not match:
            print(f"ERROR: species key '{args.only}' not found")
            print(f"Valid keys: {sorted(SPECIES.values())}")
            sys.exit(1)
        targets = match

    if args.refetch:
        # Clear existing files for targets
        for key in targets.values():
            p = OUTPUT_DIR / f"{key}.png"
            if p.exists():
                p.unlink()

    print(f"Fetching illustrations for {len(targets)} species")
    print(f"Output: {OUTPUT_DIR}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    success, failed = 0, []

    for scientific_name, species_key in targets.items():
        ok = fetch_one(scientific_name, species_key, args.dry_run)
        if ok:
            success += 1
        else:
            failed.append(species_key)
        # Be polite to Wikipedia's API
        if not args.dry_run:
            time.sleep(1.5)

    print(f"\n{'='*50}")
    print(f"Done. {success}/{len(targets)} fetched successfully.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
        print(f"\nFor missing species, manually place a PNG at:")
        for key in failed:
            print(f"  inference/illustrations/{key}.png")
    print(f"{'='*50}")

    if not args.dry_run:
        print(f"\nDeploy illustrations to Pi:")
        print(f"  rsync -avz inference/illustrations/ pi@192.168.4.73:/home/pi/forager/illustrations/")


if __name__ == "__main__":
    main()
