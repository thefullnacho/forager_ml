"""
convergence.py — Build a single ForagerResult from the two-stage router pipeline.

The domain router determines which expert to run. This module takes the
router's domain prediction and the winning expert's RawPrediction and
produces a single ForagerResult with species metadata and safety info.

Safety-first: DEADLY findings are always flagged prominently.
"""

from dataclasses import dataclass

import numpy as np

from .runner import RawPrediction


# ── Tunable parameters ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.75   # used in log_predictions() threshold marker
LOW_CONFIDENCE_THRESHOLD = 0.50  # below this -> show "LOW CONFIDENCE" in display


# ── Safety metadata ───────────────────────────────────────────────────────────
SPECIES_METADATA: dict[str, dict] = {
    # Berry expert
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

    # High-value expert
    "chanterelles_edible":            {"safety": "SAFE",    "scientific": "Cantharellus cibarius",    "lookalike": "Jack-o'-lantern",       "key_diff": "Jack-o'-lantern has true gills, grows in clusters"},
    "morels_edible":                  {"safety": "SAFE",    "scientific": "Morchella esculenta",      "lookalike": "False morel",           "key_diff": "True morel is fully hollow; false morel has cottony interior"},
    "chicken_of_the_woods":           {"safety": "SAFE",    "scientific": "Laetiporus sulphureus",    "lookalike": "N/A",                   "key_diff": "Unmistakable orange shelf; avoid on conifers"},
    "lions_mane":                     {"safety": "SAFE",    "scientific": "Hericium erinaceus",       "lookalike": "N/A",                   "key_diff": "White cascade of teeth — no true lookalike"},
    "chaga_medicinal":                {"safety": "SAFE",    "scientific": "Inonotus obliquus",        "lookalike": "Burnt wood knot",       "key_diff": "Orange-yellow interior when cut"},
    "reishi_mushroom":                {"safety": "SAFE",    "scientific": "Ganoderma lucidum",        "lookalike": "N/A",                   "key_diff": "Shiny lacquered cap, white pore surface — verify species for region"},
    "reishi_northeast":               {"safety": "SAFE",    "scientific": "Ganoderma tsugae",         "lookalike": "N/A",                   "key_diff": "Shiny lacquered cap, white pore surface"},
    "ramps_wild_leek":                {"safety": "SAFE",    "scientific": "Allium tricoccum",         "lookalike": "Lily of the valley",    "key_diff": "Lily of the valley has no garlic smell — critical check"},
    "ostrich_fern_fiddlehead":        {"safety": "CAUTION", "scientific": "Matteuccia struthiopteris","lookalike": "Bracken fern",          "key_diff": "Ostrich fern has deep U-shaped groove on stem"},
    "ginseng_american":               {"safety": "SAFE",    "scientific": "Panax quinquefolius",      "lookalike": "N/A",                   "key_diff": "Protected species — observe, don't harvest"},
    "saffron_crocus":                 {"safety": "SAFE",    "scientific": "Crocus sativus",           "lookalike": "Autumn crocus",         "key_diff": "Autumn crocus is highly toxic — 3 stigmas only in saffron"},
    "high_value_toxics":              {"safety": "DEADLY",  "scientific": "Various",                  "lookalike": "N/A",                   "key_diff": "High-value toxic lookalike class — do not consume"},

    # Psychedelics expert
    "amanita_phalloides_deadly":      {"safety": "DEADLY",  "scientific": "Amanita phalloides",       "lookalike": "Puffball (young)",      "key_diff": "Death cap has volva at base, white gills, ring"},
    "amanita_muscaria_toxic":         {"safety": "DEADLY",  "scientific": "Amanita muscaria",         "lookalike": "N/A",                   "key_diff": "Red cap with white warts — highly toxic"},
    "galerina_marginata_toxic":       {"safety": "DEADLY",  "scientific": "Galerina marginata",       "lookalike": "Psilocybe species",     "key_diff": "Rusty brown spore print; ring present — deadly lookalike"},
    "conocybe_filaris_deadly":        {"safety": "DEADLY",  "scientific": "Conocybe filaris",         "lookalike": "Psilocybe species",     "key_diff": "Rusty-brown spores; tiny ring on stem"},
    "gymnopilus_junonius":            {"safety": "CAUTION", "scientific": "Gymnopilus junonius",      "lookalike": "Chanterelle",           "key_diff": "Very bitter taste; yellow-orange gills"},
    "panaeolus_cinctulus":            {"safety": "CAUTION", "scientific": "Panaeolus cinctulus",      "lookalike": "Edible field mushrooms","key_diff": "Brown rim band on cap; dung/rich soil habitat"},
    "psilocybe_ovoideocystidiata":    {"safety": "CAUTION", "scientific": "Psilocybe ovoideocystidiata","lookalike": "Galerina marginata",  "key_diff": "Blue bruising; wood chip habitat; rusty spores in Galerina"},
    "psilocybe_cubensis":             {"safety": "CAUTION", "scientific": "Psilocybe cubensis",       "lookalike": "Galerina marginata",    "key_diff": "Bruises blue; purple-black spore print — Galerina does not bruise"},
    "psilocybe_cyanescens":           {"safety": "CAUTION", "scientific": "Psilocybe cyanescens",     "lookalike": "Galerina marginata",    "key_diff": "Wavy cap edge; strong blue bruising"},
    "psilocybe_semilanceata":         {"safety": "CAUTION", "scientific": "Psilocybe semilanceata",   "lookalike": "Conocybe filaris",      "key_diff": "Pointed nipple-cap; deep blue bruising"},
    "psilocybe_azurescens":           {"safety": "CAUTION", "scientific": "Psilocybe azurescens",     "lookalike": "Galerina marginata",    "key_diff": "Caramel cap, very potent blue bruising"},
    "psilocybe_caerulipes":           {"safety": "CAUTION", "scientific": "Psilocybe caerulipes",     "lookalike": "Galerina marginata",    "key_diff": "Blue stem base; deciduous wood debris habitat"},
    "other_mushroom":                 {"safety": "UNKNOWN", "scientific": "Unknown",                  "lookalike": "N/A",                   "key_diff": "Cannot identify — do not consume"},
    "panax_quinquefolius_ginseng_conservation": {
                                       "safety": "SAFE",    "scientific": "Panax quinquefolius",      "lookalike": "N/A",                   "key_diff": "Protected — observe only, do not harvest"},

    # Medicinals expert
    "boneset":                        {"safety": "CAUTION", "scientific": "Eupatorium perfoliatum",   "lookalike": "White snakeroot",       "key_diff": "White snakeroot has heart-shaped leaves, highly toxic — confirm perfoliate leaf pairs"},
    "burdock":                        {"safety": "SAFE",    "scientific": "Arctium lappa",            "lookalike": "Rhubarb (leaves)",      "key_diff": "Rhubarb leaves are highly toxic; burdock has burr seedheads"},
    "catnip":                         {"safety": "SAFE",    "scientific": "Nepeta cataria",           "lookalike": "N/A",                   "key_diff": "Square stem, grey-green downy leaves, minty-musty scent"},
    "coltsfoot":                      {"safety": "CAUTION", "scientific": "Tussilago farfara",        "lookalike": "Dandelion (flower)",    "key_diff": "Coltsfoot flowers appear before leaves; contains pyrrolizidine alkaloids — avoid internal use"},
    "echinacea":                      {"safety": "SAFE",    "scientific": "Echinacea purpurea",       "lookalike": "N/A",                   "key_diff": "Spiny orange-brown cone with drooping purple rays"},
    "foxglove_toxic":                 {"safety": "DEADLY",  "scientific": "Digitalis purpurea",       "lookalike": "Comfrey (rosette)",     "key_diff": "Foxglove has tubular spotted flowers; all parts highly toxic — cardiac glycosides"},
    "goldenrod":                      {"safety": "SAFE",    "scientific": "Solidago canadensis",      "lookalike": "N/A",                   "key_diff": "Arching plumes of small yellow flowers in late summer"},
    "motherwort":                     {"safety": "CAUTION", "scientific": "Leonurus cardiaca",        "lookalike": "N/A",                   "key_diff": "Square stem, deeply lobed leaves, pink-purple flowers; avoid in pregnancy"},
    "mullein":                        {"safety": "SAFE",    "scientific": "Verbascum thapsus",        "lookalike": "N/A",                   "key_diff": "Distinctive tall spike, large velvety basal rosette leaves"},
    "plantain_broadleaf":             {"safety": "SAFE",    "scientific": "Plantago major",           "lookalike": "N/A",                   "key_diff": "Oval ribbed leaves, parallel veins, narrow seedhead spike"},
    "poison_hemlock_deadly":          {"safety": "DEADLY",  "scientific": "Conium maculatum",         "lookalike": "Wild carrot",           "key_diff": "Purple-blotched hollow stem, musty smell, no hairy stem — ALL parts deadly"},
    "red_clover":                     {"safety": "SAFE",    "scientific": "Trifolium pratense",       "lookalike": "N/A",                   "key_diff": "Pink-purple globe flowers, trifoliate leaves with pale V-chevron"},
    "st_johns_wort":                  {"safety": "CAUTION", "scientific": "Hypericum perforatum",     "lookalike": "N/A",                   "key_diff": "Yellow 5-petalled flowers with black dots; translucent leaf dots; photosensitizing"},
    "stinging_nettle":                {"safety": "SAFE",    "scientific": "Urtica dioica",            "lookalike": "Wood nettle",           "key_diff": "Cook or dry to neutralize sting; serrated leaves, opposite pairs"},
    "valerian":                       {"safety": "CAUTION", "scientific": "Valeriana officinalis",    "lookalike": "Water hemlock",         "key_diff": "Water hemlock has purple-streaked hollow stem, chambered root — deadly; valerian has pinnate leaves"},
    "water_hemlock_deadly":           {"safety": "DEADLY",  "scientific": "Cicuta maculata",          "lookalike": "Wild carrot / Valerian","key_diff": "Chambered root, purple-streaked hollow stem — most violently toxic plant in NA"},
    "white_snakeroot_toxic":          {"safety": "DEADLY",  "scientific": "Ageratina altissima",      "lookalike": "Boneset",               "key_diff": "Heart-shaped leaves, flat-topped white flowers; causes milk sickness — avoid"},
    "wild_bergamot":                  {"safety": "SAFE",    "scientific": "Monarda fistulosa",        "lookalike": "N/A",                   "key_diff": "Lavender ragged flowers, square stem, oregano-like scent"},
    "wild_carrot":                    {"safety": "CAUTION", "scientific": "Daucus carota",            "lookalike": "Poison hemlock / Water hemlock", "key_diff": "Hairy stem, central purple floret, carroty smell — confirm all three before use"},
    "wood_nettle":                    {"safety": "SAFE",    "scientific": "Laportea canadensis",      "lookalike": "Stinging nettle",       "key_diff": "Alternate leaves (vs opposite in stinging nettle); forested habitat; cook to neutralize"},
    "yarrow":                         {"safety": "CAUTION", "scientific": "Achillea millefolium",     "lookalike": "Poison hemlock (leaf)", "key_diff": "Flat-topped white flower clusters, ferny aromatic leaves; hemlock has blotched hollow stem"},
}

UNKNOWN_META = {"safety": "UNKNOWN", "scientific": "Unknown", "lookalike": "N/A", "key_diff": "No confident identification"}


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class ForagerResult:
    """Single identification result from the two-stage pipeline."""
    domain:          str    # "berry" | "mushroom" | "plant" | router raw output
    species:         str    # class key, e.g. "blackberry_common"
    scientific_name: str
    confidence:      float
    safety:          str    # SAFE | CAUTION | DEADLY | UNKNOWN
    lookalike:       str
    key_diff:        str
    low_confidence:  bool   # True when confidence < LOW_CONFIDENCE_THRESHOLD
    expert_model:    str    # which expert produced this result

    @property
    def is_deadly(self) -> bool:
        return self.safety == "DEADLY" and not self.low_confidence

    @property
    def is_unknown(self) -> bool:
        return self.species == "unknown"


# ── Debug logging ─────────────────────────────────────────────────────────────

def log_predictions(domain: str, prediction: RawPrediction | None):
    """Print raw model output for debugging."""
    print("\n  -- Router + expert prediction --------------------------")
    print(f"  Domain: {domain}")
    if prediction is None:
        print("  Expert: (none — router confidence below threshold)")
    else:
        pred = prediction
        n_probs   = len(pred.probabilities)
        n_classes = len(pred.classes)
        print(f"  [{pred.model}]  output_size={n_probs}  classes={n_classes}", end="")
        if n_probs != n_classes:
            print(f"  MISMATCH", end="")
        print()
        top5_idx = np.argsort(pred.probabilities)[::-1][:5]
        for idx in top5_idx:
            label  = pred.classes[idx] if idx < n_classes else f"<unknown_idx_{idx}>"
            marker = "+" if pred.probabilities[idx] >= CONFIDENCE_THRESHOLD else "-"
            print(f"    {marker} {label:<45} {pred.probabilities[idx]:.1%}")
    print("  -------------------------------------------------------\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def build_result(domain: str, prediction: RawPrediction | None) -> ForagerResult:
    """
    Convert a domain + expert prediction into a single ForagerResult.

    If prediction is None (router confidence too low), returns an UNKNOWN result.
    """
    log_predictions(domain, prediction)

    if prediction is None:
        return ForagerResult(
            domain=domain,
            species="unknown",
            scientific_name=UNKNOWN_META["scientific"],
            confidence=0.0,
            safety=UNKNOWN_META["safety"],
            lookalike=UNKNOWN_META["lookalike"],
            key_diff=UNKNOWN_META["key_diff"],
            low_confidence=True,
            expert_model="none",
        )

    meta = SPECIES_METADATA.get(prediction.top_class, UNKNOWN_META)
    low  = prediction.top_confidence < LOW_CONFIDENCE_THRESHOLD

    return ForagerResult(
        domain=domain,
        species=prediction.top_class,
        scientific_name=meta["scientific"],
        confidence=prediction.top_confidence,
        safety=meta["safety"],
        lookalike=meta["lookalike"],
        key_diff=meta["key_diff"],
        low_confidence=low,
        expert_model=prediction.model,
    )
