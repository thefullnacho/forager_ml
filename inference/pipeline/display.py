"""
display.py — Render ForagerResult to the Waveshare 3.7" e-Paper HAT (480x280).

Driver: epd3in7 (4-gray mode for better text rendering)
Library: waveshare_epd (assumed installed on Pi at /usr/local/lib or ~/waveshare)

Layout (480 wide x 280 tall, landscape, single result):
  +--------------------------------------------------------------+
  |  FORAGER                                        [header 24px]|
  +--------------------------------------------------------------+
  |                                                              |
  |  Domain: BERRY                                               |
  |                                                              |
  |  Wild Blackberry                                             |
  |  Rubus allegheniensis                                        |
  |                                                              |
  |  Confidence: 91%                                             |
  |                                                              |
  |  Lookalike: Pokeweed (young)                                 |
  |  Pokeweed has smooth stems, white flowers                    |
  |                                                              |
  |  [             SAFE             ]   <- safety banner         |
  |                                                              |
  +--------------------------------------------------------------+

Safety colours (grayscale eInk):
  SAFE    -> LIGHT_GRAY bg / BLACK fg
  CAUTION -> DARK_GRAY bg  / WHITE fg
  DEADLY  -> BLACK bg      / WHITE fg   (maximum visibility)
  UNKNOWN -> DARK_GRAY bg  / WHITE fg
"""

from PIL import Image, ImageDraw, ImageFont

from .convergence import ForagerResult

# ── Display constants ─────────────────────────────────────────────────────────
WIDTH  = 480
HEIGHT = 280

BLACK      = 0
DARK_GRAY  = 85
LIGHT_GRAY = 170
WHITE      = 255

SAFETY_BG: dict[str, int] = {
    "SAFE":    LIGHT_GRAY,
    "CAUTION": DARK_GRAY,
    "DEADLY":  BLACK,
    "UNKNOWN": DARK_GRAY,
}
SAFETY_FG: dict[str, int] = {
    "SAFE":    BLACK,
    "CAUTION": WHITE,
    "DEADLY":  WHITE,
    "UNKNOWN": WHITE,
}
SAFETY_LABEL: dict[str, str] = {
    "SAFE":    "SAFE",
    "CAUTION": "CAUTION",
    "DEADLY":  "DO NOT EAT",
    "UNKNOWN": "UNKNOWN",
}

# Font paths
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = FONT_PATHS if not bold else [FONT_PATHS[0]] + FONT_PATHS[1:]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _truncate(draw: ImageDraw.Draw, text: str, font, max_px: int) -> str:
    """Trim text with ellipsis to fit within max_px pixels."""
    if draw.textlength(text, font=font) <= max_px:
        return text
    while text and draw.textlength(text + "...", font=font) > max_px:
        text = text[:-1]
    return text + "..."


# ── Main renderer ─────────────────────────────────────────────────────────────

def render(result: ForagerResult) -> Image.Image:
    """
    Build a 480x280 grayscale PIL image from a ForagerResult.

    Single-result layout: domain header, species name, scientific name,
    confidence, lookalike warning, and a full-width safety banner.

    Returns:
        PIL Image in 'L' mode, ready for epd3in7 4-gray display.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    font_header  = _load_font(12, bold=True)
    font_domain  = _load_font(13, bold=True)
    font_species = _load_font(22, bold=True)
    font_sci     = _load_font(14)
    font_conf    = _load_font(16, bold=True)
    font_detail  = _load_font(12)
    font_banner  = _load_font(16, bold=True)

    max_text_w = WIDTH - 20  # 10px padding each side
    x_pad = 10

    # ── Header strip ────────────────────────────────────────────────────────
    header_h = 24
    draw.rectangle([(0, 0), (WIDTH, header_h)], fill=DARK_GRAY)
    draw.text((6, 5), "FORAGER", font=font_header, fill=WHITE)

    y = header_h + 8

    if result.is_unknown and result.low_confidence:
        # Unknown domain — show a centered message
        draw.text(
            (x_pad, y + 20),
            "DOMAIN NOT RECOGNIZED",
            font=font_species,
            fill=DARK_GRAY,
        )
        draw.text(
            (x_pad, y + 50),
            "Router confidence too low. Try a clearer photo.",
            font=font_detail,
            fill=DARK_GRAY,
        )
    else:
        # ── Domain label ────────────────────────────────────────────────────
        domain_text = f"Domain: {result.domain.upper()}"
        draw.text((x_pad, y), domain_text, font=font_domain, fill=DARK_GRAY)
        y += 22

        # ── Species name ────────────────────────────────────────────────────
        if result.low_confidence:
            species_text = "LOW CONFIDENCE"
            sci_text = ""
        else:
            raw_name = result.species.replace("_", " ").title()
            species_text = _truncate(draw, raw_name, font_species, max_text_w)
            sci_text = _truncate(draw, result.scientific_name, font_sci, max_text_w)

        draw.text((x_pad, y), species_text, font=font_species, fill=BLACK)
        y += 28

        if sci_text:
            draw.text((x_pad, y), sci_text, font=font_sci, fill=DARK_GRAY)
        y += 20

        # ── Confidence ──────────────────────────────────────────────────────
        if result.low_confidence:
            conf_text = "Confidence: --"
        else:
            conf_text = f"Confidence: {int(result.confidence * 100)}%"
        draw.text((x_pad, y), conf_text, font=font_conf, fill=BLACK)
        y += 24

        # ── Lookalike warning ───────────────────────────────────────────────
        if result.lookalike and result.lookalike != "N/A" and not result.low_confidence:
            lookalike_text = f"Lookalike: {result.lookalike}"
            draw.text((x_pad, y), _truncate(draw, lookalike_text, font_detail, max_text_w), font=font_detail, fill=DARK_GRAY)
            y += 16
            if result.key_diff:
                draw.text((x_pad, y), _truncate(draw, result.key_diff, font_detail, max_text_w), font=font_detail, fill=DARK_GRAY)
            y += 16

    # ── Safety banner (always at bottom) ────────────────────────────────────
    banner_h = 36
    banner_top = HEIGHT - banner_h
    bg = SAFETY_BG.get(result.safety, DARK_GRAY)
    fg = SAFETY_FG.get(result.safety, WHITE)
    banner_label = SAFETY_LABEL.get(result.safety, "UNKNOWN")

    draw.rectangle([(0, banner_top), (WIDTH, HEIGHT)], fill=bg)
    bbox   = draw.textbbox((0, 0), banner_label, font=font_banner)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    bx = (WIDTH - text_w) // 2
    by = banner_top + (banner_h - text_h) // 2
    draw.text((bx, by), banner_label, font=font_banner, fill=fg)

    return img


# ── Display driver interface ──────────────────────────────────────────────────

class EinkDisplay:
    """
    Thin wrapper around the Waveshare epd3in7 driver.

    Keeps the display initialised between updates so refresh is faster.
    Call .show(result) to render and push. Call .clear() on exit.
    """

    def __init__(self):
        self._epd = None

    def __enter__(self):
        from waveshare_epd import epd3in7
        self._epd = epd3in7.EPD()
        self._epd.init(0)   # 0 = 4-gray mode
        self._epd.Clear(WHITE, 0)
        return self

    def __exit__(self, *_):
        if self._epd:
            self._epd.sleep()

    def show(self, result: ForagerResult):
        """Render result and push to display."""
        image = render(result)
        self._epd.display_4Gray(self._epd.getbuffer_4Gray(image))

    def clear(self):
        if self._epd:
            self._epd.Clear(WHITE, 0)
