#!/usr/bin/env python3
"""
curate_game_images.py — Produces assets for the "Can You Beat the AI?" game
on homesteaderlabs.com.

For each class in each expert, queries iNaturalist for research-grade
observations whose photos are CC-BY or CC-0 licensed. Downloads the photo,
runs ONNX inference (expert model), picks 3 lookalike-biased distractors,
resizes images, emits manifest.json + per-domain image directories.

Note: bypasses the local val/ splits because their photo IDs lack license
metadata. Fresh iNat queries give us full provenance per image.

Run:
    cd /home/alex/Documents/Forager/forager_ml
    source .venv_curation/bin/activate
    python scripts/curate_game_images.py --per-class 3 --out scripts/output

Outputs:
    scripts/output/manifest.json
    scripts/output/images/{domain}/{photo_id}.jpg
    scripts/output/inat_cache.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Paths & domain config
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "inference" / "onnx_staging"
CLASS_DIR = REPO_ROOT / "inference" / "models"

DOMAIN_CONFIG = {
    "berry": {
        "expert_onnx":   MODEL_DIR / "berry_expert_logits.onnx",
        "expert_classes": CLASS_DIR / "berry_expert_classes.json",
        "label":         "Wild Berries",
    },
    "mushroom_mycologist": {
        "expert_onnx":   MODEL_DIR / "psychedelics_expert_logits.onnx",
        "expert_classes": CLASS_DIR / "psychedelics_expert_classes.json",
        "label":         "Psilocybes & Lookalikes",
    },
    "mushroom_highvalue": {
        "expert_onnx":   MODEL_DIR / "highvalue_expert_logits.onnx",
        "expert_classes": CLASS_DIR / "highvalue_expert_classes.json",
        "label":         "Edible Mushrooms",
    },
    "medicinal": {
        "expert_onnx":   MODEL_DIR / "medicinals_expert_logits.onnx",
        "expert_classes": CLASS_DIR / "medicinals_expert_classes.json",
        "label":         "Wild Medicinals",
    },
}

# Taxon ID mappings — extracted from data/acquisition/*_pull_inat.py
# Each value is tuple of iNat taxon IDs.
TAXON_IDS = {
    # Berry
    "blackberry_common":             (82110, 82111),
    "blueberry_highbush":            (52740,),
    "blueberry_wild":                (84542, 60942),
    "elderberry_american":           (84300, 765394),
    "staghorn_sumac":                (167829,),
    "wild_grape_riverbank":          (119936,),
    "bittersweet_nightshade_toxic":  (55620,),
    "canada_moonseed_deadly":        (130900,),
    "poison_ivy":                    (58732,),
    "pokeweed_toxic":                (48599,),
    "virginia_creeper_toxic":        (50278,),

    # High-value
    "chanterelles_edible":           (47347,),
    "morels_edible":                 (56830,),
    "lions_mane":                    (49158,),
    "chicken_of_the_woods":          (48431,),
    "chaga_medicinal":               (127021,),
    "reishi_northeast":              (63274, 350824),
    "ginseng_american":              (116333,),
    "ramps_wild_leek":               (55634,),
    "ostrich_fern_fiddlehead":       (82574,),
    "saffron_crocus":                (118933,),
    "high_value_toxics":             (85120, 52135, 67356, 125390, 154735, 58694),

    # Psychedelics (mushroom_mycologist domain)
    "psilocybe_cubensis":            (328244,),
    "psilocybe_cyanescens":          (179085,),
    "psilocybe_semilanceata":        (54025,),
    "psilocybe_caerulipes":          (206899,),
    "psilocybe_azurescens":          (179652,),
    "gymnopilus_junonius":           (83196,),
    "galerina_marginata_toxic":      (154735,),
    "amanita_muscaria_toxic":        (48715,),
    "amanita_phalloides_deadly":     (52135,),
    "conocybe_filaris_deadly":       (877494,),
    "panax_quinquefolius_ginseng_conservation": (116333,),
    "other_mushroom":                (47170,),

    # Medicinals
    "stinging_nettle":      (51884,),
    "wood_nettle":          (119802,),
    "yarrow":               (52821,),
    "mullein":              (59029,),
    "goldenrod":            (67808,),
    "st_johns_wort":        (56077,),
    "plantain_broadleaf":   (58961,),
    "burdock":              (59570,),
    "boneset":              (119045,),
    "red_clover":           (51875,),
    "wild_bergamot":        (85320,),
    "catnip":               (48622,),
    "motherwort":           (56171,),
    "valerian":             (56160,),
    "echinacea":            (48627,),
    "coltsfoot":            (56222,),
    "wild_carrot":          (76610,),
    "poison_hemlock_deadly":(52998,),
    "water_hemlock_deadly": (60125,),
    "white_snakeroot_toxic":(119048,),
    "foxglove_toxic":       (53983,),
}


def class_to_label(cls: str) -> str:
    pretty = (
        cls.replace("_toxic", "")
           .replace("_deadly", "")
           .replace("_edible", "")
           .replace("_medicinal", "")
    )
    return pretty.replace("_", " ").title()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

INAT_API_BASE = "https://api.inaturalist.org/v1"
ALLOWED_LICENSES = {"cc-by", "cc0"}
RATE_LIMIT_S = 1.0


# ============================================================================
# Species metadata — mirrors inference/pipeline/convergence.py
# ============================================================================

SPECIES_METADATA = {
    "blackberry_common":              {"safety": "SAFE",    "scientific": "Rubus allegheniensis",     "lookalike": "Pokeweed (young)",      "key_diff": "Pokeweed has smooth stems, white flowers"},
    "blueberry_highbush":             {"safety": "SAFE",    "scientific": "Vaccinium corymbosum",     "lookalike": "Canada moonseed",       "key_diff": "Moonseed has one crescent seed, no true drupelets"},
    "blueberry_wild":                 {"safety": "SAFE",    "scientific": "Vaccinium angustifolium",  "lookalike": "Canada moonseed",       "key_diff": "Moonseed has one crescent seed, no true drupelets"},
    "elderberry_american":            {"safety": "CAUTION", "scientific": "Sambucus canadensis",      "lookalike": "Pokeweed",              "key_diff": "Elderberry has compound leaves; must be cooked"},
    "staghorn_sumac":                 {"safety": "SAFE",    "scientific": "Rhus typhina",             "lookalike": "Poison sumac",          "key_diff": "Poison sumac has white berries, swampy habitat"},
    "wild_grape_riverbank":           {"safety": "SAFE",    "scientific": "Vitis riparia",            "lookalike": "Canada moonseed",       "key_diff": "Grape has tendrils and true seeds"},
    "bittersweet_nightshade_toxic":   {"safety": "DEADLY",  "scientific": "Solanum dulcamara",        "lookalike": "N/A",                   "key_diff": "Purple flowers, red-to-black berries — avoid"},
    "canada_moonseed_deadly":         {"safety": "DEADLY",  "scientific": "Menispermum canadense",    "lookalike": "Wild grape",            "key_diff": "Crescent-shaped seed, no tendrils"},
    "poison_ivy":                     {"safety": "DEADLY",  "scientific": "Toxicodendron radicans",   "lookalike": "N/A",                   "key_diff": "Leaves of three, let it be"},
    "pokeweed_toxic":                 {"safety": "DEADLY",  "scientific": "Phytolacca americana",     "lookalike": "Elderberry",            "key_diff": "Pink-red stems, hollow; all parts toxic"},
    "virginia_creeper_toxic":         {"safety": "CAUTION", "scientific": "Parthenocissus quinquefolia","lookalike": "N/A",                  "key_diff": "5-leaflet vine; berries toxic"},
    "chanterelles_edible":            {"safety": "SAFE",    "scientific": "Cantharellus cibarius",    "lookalike": "Jack-o'-lantern",       "key_diff": "Jack-o'-lantern has true gills, grows in clusters"},
    "morels_edible":                  {"safety": "SAFE",    "scientific": "Morchella esculenta",      "lookalike": "False morel",           "key_diff": "True morel is fully hollow; false morel has cottony interior"},
    "chicken_of_the_woods":           {"safety": "SAFE",    "scientific": "Laetiporus sulphureus",    "lookalike": "N/A",                   "key_diff": "Unmistakable orange shelf; avoid on conifers"},
    "lions_mane":                     {"safety": "SAFE",    "scientific": "Hericium erinaceus",       "lookalike": "N/A",                   "key_diff": "White cascade of teeth — no true lookalike"},
    "chaga_medicinal":                {"safety": "SAFE",    "scientific": "Inonotus obliquus",        "lookalike": "Burnt wood knot",       "key_diff": "Orange-yellow interior when cut"},
    "reishi_northeast":               {"safety": "SAFE",    "scientific": "Ganoderma tsugae",         "lookalike": "N/A",                   "key_diff": "Shiny lacquered cap, white pore surface"},
    "ramps_wild_leek":                {"safety": "SAFE",    "scientific": "Allium tricoccum",         "lookalike": "Lily of the valley",    "key_diff": "Lily of the valley has no garlic smell — critical check"},
    "ostrich_fern_fiddlehead":        {"safety": "CAUTION", "scientific": "Matteuccia struthiopteris","lookalike": "Bracken fern",          "key_diff": "Ostrich fern has deep U-shaped groove on stem"},
    "ginseng_american":               {"safety": "SAFE",    "scientific": "Panax quinquefolius",      "lookalike": "N/A",                   "key_diff": "Protected species — observe, don't harvest"},
    "saffron_crocus":                 {"safety": "SAFE",    "scientific": "Crocus sativus",           "lookalike": "Autumn crocus",         "key_diff": "Autumn crocus is highly toxic — 3 stigmas only in saffron"},
    "high_value_toxics":              {"safety": "DEADLY",  "scientific": "Various",                  "lookalike": "N/A",                   "key_diff": "High-value toxic lookalike class — do not consume"},
    "amanita_phalloides_deadly":      {"safety": "DEADLY",  "scientific": "Amanita phalloides",       "lookalike": "Puffball (young)",      "key_diff": "Death cap has volva at base, white gills, ring"},
    "amanita_muscaria_toxic":         {"safety": "DEADLY",  "scientific": "Amanita muscaria",         "lookalike": "N/A",                   "key_diff": "Red cap with white warts — highly toxic"},
    "galerina_marginata_toxic":       {"safety": "DEADLY",  "scientific": "Galerina marginata",       "lookalike": "Psilocybe species",     "key_diff": "Rusty brown spore print; ring present — deadly lookalike"},
    "conocybe_filaris_deadly":        {"safety": "DEADLY",  "scientific": "Conocybe filaris",         "lookalike": "Psilocybe species",     "key_diff": "Rusty-brown spores; tiny ring on stem"},
    "gymnopilus_junonius":            {"safety": "CAUTION", "scientific": "Gymnopilus junonius",      "lookalike": "Chanterelle",           "key_diff": "Very bitter taste; yellow-orange gills"},
    "psilocybe_cubensis":             {"safety": "CAUTION", "scientific": "Psilocybe cubensis",       "lookalike": "Galerina marginata",    "key_diff": "Bruises blue; purple-black spore print — Galerina does not bruise"},
    "psilocybe_cyanescens":           {"safety": "CAUTION", "scientific": "Psilocybe cyanescens",     "lookalike": "Galerina marginata",    "key_diff": "Wavy cap edge; strong blue bruising"},
    "psilocybe_semilanceata":         {"safety": "CAUTION", "scientific": "Psilocybe semilanceata",   "lookalike": "Conocybe filaris",      "key_diff": "Pointed nipple-cap; deep blue bruising"},
    "psilocybe_azurescens":           {"safety": "CAUTION", "scientific": "Psilocybe azurescens",     "lookalike": "Galerina marginata",    "key_diff": "Caramel cap, very potent blue bruising"},
    "psilocybe_caerulipes":           {"safety": "CAUTION", "scientific": "Psilocybe caerulipes",     "lookalike": "Galerina marginata",    "key_diff": "Blue stem base; deciduous wood debris habitat"},
    "other_mushroom":                 {"safety": "UNKNOWN", "scientific": "Unknown",                  "lookalike": "N/A",                   "key_diff": "Cannot identify — do not consume"},
    "panax_quinquefolius_ginseng_conservation": {"safety": "SAFE", "scientific": "Panax quinquefolius", "lookalike": "N/A", "key_diff": "Protected — observe only, do not harvest"},
    "boneset":              {"safety": "CAUTION", "scientific": "Eupatorium perfoliatum",   "lookalike": "White snakeroot",       "key_diff": "White snakeroot has heart-shaped leaves, highly toxic"},
    "burdock":              {"safety": "SAFE",    "scientific": "Arctium lappa",            "lookalike": "Rhubarb (leaves)",      "key_diff": "Rhubarb leaves are highly toxic; burdock has burr seedheads"},
    "catnip":               {"safety": "SAFE",    "scientific": "Nepeta cataria",           "lookalike": "N/A",                   "key_diff": "Square stem, grey-green downy leaves, minty-musty scent"},
    "coltsfoot":            {"safety": "CAUTION", "scientific": "Tussilago farfara",        "lookalike": "Dandelion (flower)",    "key_diff": "Coltsfoot flowers appear before leaves; contains pyrrolizidine alkaloids"},
    "echinacea":            {"safety": "SAFE",    "scientific": "Echinacea purpurea",       "lookalike": "N/A",                   "key_diff": "Spiny orange-brown cone with drooping purple rays"},
    "foxglove_toxic":       {"safety": "DEADLY",  "scientific": "Digitalis purpurea",       "lookalike": "Comfrey (rosette)",     "key_diff": "Tubular spotted flowers; all parts highly toxic — cardiac glycosides"},
    "goldenrod":            {"safety": "SAFE",    "scientific": "Solidago canadensis",      "lookalike": "N/A",                   "key_diff": "Arching plumes of small yellow flowers in late summer"},
    "motherwort":           {"safety": "CAUTION", "scientific": "Leonurus cardiaca",        "lookalike": "N/A",                   "key_diff": "Square stem, deeply lobed leaves, pink-purple flowers"},
    "mullein":              {"safety": "SAFE",    "scientific": "Verbascum thapsus",        "lookalike": "N/A",                   "key_diff": "Distinctive tall spike, large velvety basal rosette leaves"},
    "plantain_broadleaf":   {"safety": "SAFE",    "scientific": "Plantago major",           "lookalike": "N/A",                   "key_diff": "Oval ribbed leaves, parallel veins, narrow seedhead spike"},
    "poison_hemlock_deadly":{"safety": "DEADLY",  "scientific": "Conium maculatum",         "lookalike": "Wild carrot",           "key_diff": "Purple-blotched hollow stem, musty smell — ALL parts deadly"},
    "red_clover":           {"safety": "SAFE",    "scientific": "Trifolium pratense",       "lookalike": "N/A",                   "key_diff": "Pink-purple globe flowers, trifoliate leaves with pale V-chevron"},
    "st_johns_wort":        {"safety": "CAUTION", "scientific": "Hypericum perforatum",     "lookalike": "N/A",                   "key_diff": "Yellow 5-petalled flowers with black dots; photosensitizing"},
    "stinging_nettle":      {"safety": "SAFE",    "scientific": "Urtica dioica",            "lookalike": "Wood nettle",           "key_diff": "Cook or dry to neutralize sting; serrated leaves, opposite pairs"},
    "valerian":             {"safety": "CAUTION", "scientific": "Valeriana officinalis",    "lookalike": "Water hemlock",         "key_diff": "Water hemlock has purple-streaked hollow stem; valerian has pinnate leaves"},
    "water_hemlock_deadly": {"safety": "DEADLY",  "scientific": "Cicuta maculata",          "lookalike": "Wild carrot / Valerian","key_diff": "Chambered root, purple-streaked hollow stem — most violently toxic plant in NA"},
    "white_snakeroot_toxic":{"safety": "DEADLY",  "scientific": "Ageratina altissima",      "lookalike": "Boneset",               "key_diff": "Heart-shaped leaves, flat-topped white flowers; causes milk sickness"},
    "wild_bergamot":        {"safety": "SAFE",    "scientific": "Monarda fistulosa",        "lookalike": "N/A",                   "key_diff": "Lavender ragged flowers, square stem, oregano-like scent"},
    "wild_carrot":          {"safety": "CAUTION", "scientific": "Daucus carota",            "lookalike": "Poison hemlock",        "key_diff": "Hairy stem, central purple floret, carroty smell"},
    "wood_nettle":          {"safety": "SAFE",    "scientific": "Laportea canadensis",      "lookalike": "Stinging nettle",       "key_diff": "Alternate leaves (vs opposite in stinging nettle); forested habitat"},
    "yarrow":               {"safety": "CAUTION", "scientific": "Achillea millefolium",     "lookalike": "Poison hemlock (leaf)", "key_diff": "Flat-topped white flower clusters, ferny aromatic leaves"},
}

UNKNOWN_META = {"safety": "UNKNOWN", "scientific": "Unknown", "lookalike": "N/A", "key_diff": "No documented metadata"}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class CandidateImage:
    domain: str
    true_class: str
    photo_id: str
    observation_id: str
    photo_url: str
    license_code: str
    observer: str
    observation_url: str
    local_path: Optional[Path] = None


@dataclass
class CuratedRound:
    domain: str
    photo_id: str
    true_class: str
    options: list[str]
    ai_predicted_class: str
    ai_confidence: float
    top_probs: dict[str, float]
    ai_correct: bool
    license_code: str
    observer: str
    observation_url: str


# ============================================================================
# iNat client
# ============================================================================

class InatClient:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: dict = {}
        if cache_path.exists():
            try: self.cache = json.loads(cache_path.read_text())
            except: self.cache = {}
        self.last_call_ts = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "HomesteaderLabs/forager-game-curation/1.0"

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_ts
        if elapsed < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - elapsed)
        self.last_call_ts = time.time()

    def search_observations(self, taxon_ids: tuple[int, ...], per_page: int = 30) -> list[dict]:
        """Returns research-grade observations filtered to CC-BY/CC-0 photos."""
        cache_key = f"search:{','.join(map(str, taxon_ids))}:{per_page}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        try:
            r = self.session.get(
                f"{INAT_API_BASE}/observations",
                params={
                    "taxon_id": ",".join(map(str, taxon_ids)),
                    "quality_grade": "research",
                    "photo_license": "cc-by,cc0",
                    "per_page": per_page,
                    "order": "desc",
                    "order_by": "observed_on",
                },
                timeout=15,
            )
        except Exception as e:
            print(f"  [inat] error: {e}", file=sys.stderr)
            return []

        if r.status_code != 200:
            print(f"  [inat] HTTP {r.status_code}", file=sys.stderr)
            return []

        results = r.json().get("results", [])
        self.cache[cache_key] = results
        self._flush()
        return results

    def _flush(self):
        self.cache_path.write_text(json.dumps(self.cache, indent=2))


# ============================================================================
# Image download
# ============================================================================

def normalize_photo_url(url: str) -> str:
    """Convert a 'square' or 'thumb' iNat URL to 'medium' (~500px wide)."""
    for size in ("square", "thumb", "small", "original"):
        url = url.replace(f"/{size}.", "/medium.")
    return url


def download_photo(url: str, dst: Path) -> bool:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "HomesteaderLabs/forager-game-curation/1.0"})
        if r.status_code != 200:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.content)
        return True
    except Exception:
        return False


# ============================================================================
# ONNX inference
# ============================================================================

def load_onnx(path: Path) -> ort.InferenceSession:
    if not path.exists(): raise FileNotFoundError(f"ONNX not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_classes(path: Path) -> list[str]:
    return json.loads(path.read_text())["classes"]


def preprocess(image_path: Path) -> np.ndarray:
    im = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


# ============================================================================
# Distractor selection
# ============================================================================

def pick_distractors(true_class: str, all_classes: list[str], rng: random.Random) -> list[str]:
    meta = SPECIES_METADATA.get(true_class, UNKNOWN_META)
    lookalike_str = (meta.get("lookalike") or "").lower()

    preferred: list[str] = []
    for c in all_classes:
        if c == true_class: continue
        core = c.split("_")[0]
        if core and core in lookalike_str:
            preferred.append(c)

    pool = [c for c in all_classes if c != true_class]
    rng.shuffle(pool)
    picks: list[str] = []
    for c in preferred[:2]:
        if c not in picks: picks.append(c)
    for c in pool:
        if len(picks) >= 3: break
        if c not in picks: picks.append(c)
    while len(picks) < 3 and pool:
        picks.append(pool.pop())
    return picks


# ============================================================================
# Resize
# ============================================================================

def resize_and_save(src: Path, dst: Path, max_edge: int = 800, quality: int = 85) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality, optimize=True)


# ============================================================================
# Domain-class mapping
# ============================================================================

DOMAIN_CLASSES = {}  # filled at runtime once we load class JSONs


def class_to_domain(cls: str) -> Optional[str]:
    for dom, classes in DOMAIN_CLASSES.items():
        if cls in classes: return dom
    return None


# ============================================================================
# Main
# ============================================================================

def harvest_class(
    cls: str,
    domain: str,
    per_class: int,
    inat: InatClient,
    images_root: Path,
    raw_root: Path,
) -> list[CandidateImage]:
    taxa = TAXON_IDS.get(cls)
    if not taxa:
        print(f"  [skip] no taxon for {cls}", file=sys.stderr)
        return []

    observations = inat.search_observations(taxa, per_page=30)
    selected: list[CandidateImage] = []

    for obs in observations:
        if len(selected) >= per_class: break
        # Pick first CC-BY/CC-0 photo from this observation
        for photo in obs.get("photos", []):
            lic = (photo.get("license_code") or "").lower()
            if lic not in ALLOWED_LICENSES: continue
            url = normalize_photo_url(photo.get("url", ""))
            if not url: continue
            photo_id = str(photo["id"])
            raw_path = raw_root / domain / f"{photo_id}.jpg"
            if not raw_path.exists():
                if not download_photo(url, raw_path):
                    continue
            observer = (obs.get("user") or {}).get("login", "unknown")
            obs_url = obs.get("uri") or f"https://www.inaturalist.org/observations/{obs['id']}"
            selected.append(CandidateImage(
                domain=domain,
                true_class=cls,
                photo_id=photo_id,
                observation_id=str(obs.get("id", "")),
                photo_url=url,
                license_code=lic,
                observer=observer,
                observation_url=obs_url,
                local_path=raw_path,
            ))
            break  # one photo per observation

    return selected


def emit_manifest(by_domain: dict[str, list[CuratedRound]], out_root: Path) -> dict:
    manifest = {
        "version": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modelMeta": {
            "architecture": "EfficientNet Lite2",
            "device": "Hailo 8L (4 TOPS int8)",
            "deviceTimingMs": 187,
            "browserTimingMs": 540,
        },
        "domains": {},
    }
    for domain, rounds in by_domain.items():
        cfg = DOMAIN_CONFIG[domain]
        manifest["domains"][domain] = {
            "label": cfg["label"],
            "rounds": [
                {
                    "id": r.photo_id,
                    "imageUrl": f"/forager-game/images/{domain}/{r.photo_id}.jpg",
                    "trueClass": r.true_class,
                    "trueLabel": class_to_label(r.true_class),
                    "options": r.options,
                    "optionLabels": { o: class_to_label(o) for o in r.options },
                    "ai": {
                        "predictedClass": r.ai_predicted_class,
                        "confidence":     round(r.ai_confidence, 4),
                        "topProbs":       { k: round(v, 4) for k, v in r.top_probs.items() },
                        "correct":        r.ai_correct,
                    },
                    "metadata": {
                        "safety":     SPECIES_METADATA.get(r.true_class, UNKNOWN_META)["safety"],
                        "scientific": SPECIES_METADATA.get(r.true_class, UNKNOWN_META)["scientific"],
                        "lookalike":  SPECIES_METADATA.get(r.true_class, UNKNOWN_META)["lookalike"],
                        "keyDiff":    SPECIES_METADATA.get(r.true_class, UNKNOWN_META)["key_diff"],
                    },
                    "attribution": {
                        "observer":       r.observer,
                        "license":        r.license_code,
                        "observationUrl": r.observation_url,
                    },
                }
                for r in rounds
            ],
        }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--per-class", type=int, default=2, help="rounds per class (across the class's domain)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", nargs="*", default=None, help="subset of domains to process")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root / "raw"
    images_root = out_root / "images"
    inat = InatClient(out_root / "inat_cache.json")

    # Load expert class lists + classify by domain
    global DOMAIN_CLASSES
    DOMAIN_CLASSES = {dom: load_classes(cfg["expert_classes"]) for dom, cfg in DOMAIN_CONFIG.items()}

    domains = args.domains or list(DOMAIN_CONFIG.keys())
    by_domain: dict[str, list[CuratedRound]] = {}

    for domain in domains:
        if domain not in DOMAIN_CONFIG:
            print(f"  [skip] unknown domain {domain}", file=sys.stderr)
            continue
        cfg = DOMAIN_CONFIG[domain]
        print(f"\n=== {domain} ({cfg['label']}) ===")

        classes = DOMAIN_CLASSES[domain]
        all_candidates: list[CandidateImage] = []
        for cls in tqdm(classes, desc=f"  harvest {domain}"):
            cands = harvest_class(cls, domain, args.per_class, inat, images_root, raw_root)
            all_candidates.extend(cands)

        print(f"  candidates: {len(all_candidates)}")
        if not all_candidates:
            continue

        expert_sess = load_onnx(cfg["expert_onnx"])
        rounds: list[CuratedRound] = []

        for cand in tqdm(all_candidates, desc=f"  infer {domain}"):
            try:
                x = preprocess(cand.local_path)
            except Exception as e:
                print(f"  [skip] preprocess {cand.photo_id}: {e}", file=sys.stderr)
                continue
            probs = softmax(expert_sess.run(None, {expert_sess.get_inputs()[0].name: x})[0], axis=-1)[0]
            distractors = pick_distractors(cand.true_class, classes, rng)
            options = [cand.true_class] + distractors
            rng.shuffle(options)
            option_indices = [classes.index(o) for o in options]
            restricted = probs[option_indices]
            restricted = restricted / (restricted.sum() + 1e-9)
            pred_idx = int(np.argmax(restricted))
            ai_pred = options[pred_idx]
            ai_conf = float(restricted[pred_idx])
            top_probs = { options[i]: float(restricted[i]) for i in range(len(options)) }

            rounds.append(CuratedRound(
                domain=domain,
                photo_id=cand.photo_id,
                true_class=cand.true_class,
                options=options,
                ai_predicted_class=ai_pred,
                ai_confidence=ai_conf,
                top_probs=top_probs,
                ai_correct=(ai_pred == cand.true_class),
                license_code=cand.license_code,
                observer=cand.observer,
                observation_url=cand.observation_url,
            ))

            # Copy + resize image
            dst = images_root / domain / f"{cand.photo_id}.jpg"
            try: resize_and_save(cand.local_path, dst)
            except Exception as e: print(f"  [skip] resize {cand.photo_id}: {e}", file=sys.stderr)

        ai_correct = sum(1 for r in rounds if r.ai_correct)
        print(f"  rounds: {len(rounds)} · AI accuracy: {ai_correct}/{len(rounds)} = {ai_correct/max(1,len(rounds))*100:.1f}%")
        by_domain[domain] = rounds

    manifest = emit_manifest(by_domain, out_root)
    total = sum(len(v) for v in by_domain.values())
    print(f"\nDone. {total} rounds across {len(by_domain)} domains → {out_root}/manifest.json")


if __name__ == "__main__":
    main()
