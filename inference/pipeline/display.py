"""
display.py — Render ForagerResult to the Waveshare 3.7" e-Paper HAT (480x280).

Driver: epd3in7 (4-gray mode for identifications, 1-bit mode for fast states)
Library: waveshare_epd (assumed installed on Pi at /usr/local/lib or ~/waveshare)

Refresh strategy:
  Scanning state  → 1-bit fast refresh (~0.3s)  — shown while inference runs
  Abstention      → 1-bit fast refresh (~0.3s)  — reposition / not a target
  Identification  → 4-gray full refresh (~2.5s) — worth the wait for readability

Layout (480 wide x 280 tall, landscape):
  +--------------------------------------------------------------+
  | FORAGER                                        [header 24px] |
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
  +------+-------------------------------------------------------+
  |SAFE  |  [safety banner 36px]                                 |
  +------+-------------------------------------------------------+

Safety colours (grayscale eInk):
  SAFE    -> LIGHT_GRAY bg / BLACK fg
  CAUTION -> DARK_GRAY bg  / WHITE fg
  DEADLY  -> BLACK bg      / WHITE fg   (maximum visibility)
  UNKNOWN -> DARK_GRAY bg  / WHITE fg
"""

import os

from PIL import Image, ImageDraw, ImageFont

from .convergence import ForagerResult

# ── Display constants ──────────────────────────────────────────────────────────
WIDTH  = 480
HEIGHT = 280

BLACK      = 0
DARK_GRAY  = 85
LIGHT_GRAY = 170
WHITE      = 255

HEADER_H = 24
BANNER_H = 36
CONTENT_TOP = HEADER_H + 8
CONTENT_BOT = HEIGHT - BANNER_H
X_PAD = 12

# Illustration zone (right side of content area)
ILLUS_W    = 184
ILLUS_H    = CONTENT_BOT - CONTENT_TOP   # full content height
ILLUS_X    = WIDTH - ILLUS_W             # right-aligned
TEXT_MAX_W = ILLUS_X - X_PAD - 4        # text zone width

ILLUSTRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "illustrations")

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
    "SAFE":    "SAFE TO EAT",
    "CAUTION": "USE CAUTION",
    "DEADLY":  "DO NOT EAT — DEADLY",
    "UNKNOWN": "UNKNOWN",
}

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


# ── Font helpers ───────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [FONT_PATHS[0]] + FONT_PATHS[1:] if bold else FONT_PATHS
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


def _centered_text(draw: ImageDraw.Draw, text: str, font, y: int, fill: int):
    """Draw text horizontally centered on the display."""
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (WIDTH - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _load_illustration(species_key: str) -> Image.Image | None:
    """
    Load a pre-processed illustration PNG for the species.
    Returns None if not found — caller renders text-only layout.
    """
    path = os.path.join(ILLUSTRATIONS_DIR, f"{species_key}.png")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("L")
        # Resize to fit the zone if needed (should already be correct size)
        if img.size != (ILLUS_W, ILLUS_H):
            img = img.resize((ILLUS_W, ILLUS_H), Image.LANCZOS)
        return img
    except Exception:
        return None


def _draw_header(draw: ImageDraw.Draw, subtitle: str = ""):
    """Draw the FORAGER header strip."""
    draw.rectangle([(0, 0), (WIDTH, HEADER_H)], fill=DARK_GRAY)
    font = _load_font(12, bold=True)
    draw.text((6, 5), "FORAGER", font=font, fill=WHITE)
    if subtitle:
        bbox = draw.textbbox((0, 0), subtitle, font=font)
        draw.text((WIDTH - bbox[2] - 8, 5), subtitle, font=font, fill=LIGHT_GRAY)


def _draw_banner(draw: ImageDraw.Draw, safety: str):
    """Draw the safety banner at the bottom."""
    bg = SAFETY_BG.get(safety, DARK_GRAY)
    fg = SAFETY_FG.get(safety, WHITE)
    label = SAFETY_LABEL.get(safety, "UNKNOWN")
    font = _load_font(15, bold=True)

    banner_top = HEIGHT - BANNER_H
    draw.rectangle([(0, banner_top), (WIDTH, HEIGHT)], fill=bg)
    bbox = draw.textbbox((0, 0), label, font=font)
    bx = (WIDTH - (bbox[2] - bbox[0])) // 2
    by = banner_top + (BANNER_H - (bbox[3] - bbox[1])) // 2
    draw.text((bx, by), label, font=font, fill=fg)


# ── State renders ──────────────────────────────────────────────────────────────

def render_scanning() -> Image.Image:
    """
    Fast 1-bit image shown while inference is running.
    Displayed via 1-bit partial refresh — appears almost instantly.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    _draw_header(draw, subtitle="SCANNING")

    font_large = _load_font(28, bold=True)
    font_small = _load_font(13)

    _centered_text(draw, "SCANNING...", font_large, 80, DARK_GRAY)
    _centered_text(draw, "Hold camera 4–6\" from subject", font_small, 128, DARK_GRAY)

    # Simple progress bar outline as a visual hint
    bar_x, bar_y, bar_w, bar_h = X_PAD, 155, WIDTH - X_PAD * 2, 8
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], outline=DARK_GRAY, width=1)

    draw.rectangle([(0, HEIGHT - BANNER_H), (WIDTH, HEIGHT)], fill=DARK_GRAY)
    _centered_text(draw, "IDENTIFYING...", _load_font(14, bold=True), HEIGHT - BANNER_H + 10, WHITE)

    return img


def render_abstention(result: ForagerResult) -> Image.Image:
    """
    Fast 1-bit image for abstentions — router didn't commit.
    Two variants based on why:
      domain == "other"  → not a foraging target at all
      domain != "other"  → right kind of subject, bad angle/distance
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    _draw_header(draw)

    font_large = _load_font(22, bold=True)
    font_small = _load_font(13)
    font_hint  = _load_font(12)

    if result.domain == "other":
        _centered_text(draw, "NOT A FORAGING TARGET", font_large, 70, DARK_GRAY)
        _centered_text(draw, "Point camera at a berry, mushroom,", font_small, 110, DARK_GRAY)
        _centered_text(draw, "or plant.", font_small, 128, DARK_GRAY)
    else:
        _centered_text(draw, "MOVE CLOSER", font_large, 70, DARK_GRAY)
        _centered_text(draw, f"Router saw: {result.domain.upper()} — but confidence too low", font_hint, 108, DARK_GRAY)
        _centered_text(draw, "Hold camera 4–6\" from subject", font_small, 126, DARK_GRAY)
        _centered_text(draw, "and ensure good lighting.", font_small, 144, DARK_GRAY)

    draw.rectangle([(0, HEIGHT - BANNER_H), (WIDTH, HEIGHT)], fill=DARK_GRAY)
    _centered_text(draw, "TRY AGAIN", _load_font(15, bold=True), HEIGHT - BANNER_H + 10, WHITE)

    return img


# ── Full identification render ─────────────────────────────────────────────────

def render(result: ForagerResult) -> Image.Image:
    """
    Full 4-gray render for a committed identification result.

    If an illustration exists for the species it is placed on the right side
    and text is constrained to the left zone. Falls back to full-width text
    layout when no illustration is available.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    font_species = _load_font(20, bold=True)
    font_sci     = _load_font(12)
    font_conf    = _load_font(14, bold=True)
    font_detail  = _load_font(11)

    _draw_header(draw, subtitle=result.domain.upper())

    # ── Illustration (right zone) ─────────────────────────────────────────────
    illus = None if result.low_confidence else _load_illustration(result.species)

    if illus is not None:
        # Paste illustration into the right zone
        img.paste(illus, (ILLUS_X, CONTENT_TOP))
        # Thin separator line between text and illustration
        draw.line([(ILLUS_X - 2, CONTENT_TOP), (ILLUS_X - 2, CONTENT_BOT)], fill=LIGHT_GRAY, width=1)
        text_w = TEXT_MAX_W
    else:
        text_w = WIDTH - X_PAD * 2

    y = CONTENT_TOP

    # ── Species name & details ────────────────────────────────────────────────
    if result.low_confidence:
        draw.text((X_PAD, y), "LOW CONFIDENCE", font=font_species, fill=DARK_GRAY)
        y += 28
        draw.text((X_PAD, y), "Ensure good lighting", font=font_detail, fill=DARK_GRAY)
        y += 15
        draw.text((X_PAD, y), "and a clear view.", font=font_detail, fill=DARK_GRAY)
    else:
        raw_name = result.species.replace("_", " ").title()
        # Strip safety suffixes — the banner communicates that
        display_name = raw_name.replace(" Toxic", "").replace(" Deadly", "").strip()
        draw.text((X_PAD, y), _truncate(draw, display_name, font_species, text_w), font=font_species, fill=BLACK)
        y += 26

        draw.text((X_PAD, y), _truncate(draw, result.scientific_name, font_sci, text_w), font=font_sci, fill=DARK_GRAY)
        y += 18

        draw.text((X_PAD, y), f"Confidence: {int(result.confidence * 100)}%", font=font_conf, fill=BLACK)
        y += 20

        # ── Lookalike warning ─────────────────────────────────────────────────
        if result.lookalike and result.lookalike != "N/A":
            draw.line([(X_PAD, y), (X_PAD + text_w, y)], fill=LIGHT_GRAY, width=1)
            y += 5
            lk = _truncate(draw, f"Lookalike: {result.lookalike}", font_detail, text_w)
            draw.text((X_PAD, y), lk, font=font_detail, fill=DARK_GRAY)
            y += 14
            if result.key_diff:
                kd = _truncate(draw, result.key_diff, font_detail, text_w)
                draw.text((X_PAD, y), kd, font=font_detail, fill=DARK_GRAY)

    _draw_banner(draw, result.safety)
    return img


# ── Display driver interface ───────────────────────────────────────────────────

class EinkDisplay:
    """
    Thin wrapper around the Waveshare epd3in7 driver.

    show_scanning() — fast 1-bit update shown while inference runs
    show(result)    — smart dispatch: 1-bit for abstentions, 4-gray for IDs
    """

    def __init__(self):
        self._epd = None
        self._mode = None   # track current mode to avoid redundant re-inits

    def __enter__(self):
        from waveshare_epd import epd3in7
        self._epd = epd3in7.EPD()
        self._set_mode(4)
        self._epd.Clear(WHITE, 0)
        return self

    def __exit__(self, *_):
        if self._epd:
            self._epd.sleep()

    def show_scanning(self):
        """Fast 1-bit update — call immediately after trigger, before inference."""
        img = render_scanning()
        self._set_mode(1)
        self._epd.display_1Gray(self._epd.getbuffer(img))

    def show(self, result: ForagerResult):
        """Render result. Uses 1-bit for abstentions, 4-gray for identifications."""
        if result.is_unknown:
            img = render_abstention(result)
            self._set_mode(1)
            self._epd.display_1Gray(self._epd.getbuffer(img))
        else:
            img = render(result)
            self._set_mode(4)
            self._epd.display_4Gray(self._epd.getbuffer_4Gray(img))

    def clear(self):
        if self._epd:
            self._set_mode(4)
            self._epd.Clear(WHITE, 0)

    def _set_mode(self, mode: int):
        """Switch between 1-bit (mode=1) and 4-gray (mode=0) only when needed."""
        epd_mode = 0 if mode == 4 else 1
        if self._mode != mode:
            self._epd.init(epd_mode)
            self._mode = mode
