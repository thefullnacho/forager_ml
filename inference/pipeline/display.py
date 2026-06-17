"""
display.py — Render ForagerResult to the Waveshare 3.7" e-Paper HAT (280x480).

Driver: epd3in7 (4-gray mode for boot splash, 1-bit mode for everything else)
Library: waveshare_epd (assumed installed on Pi at /usr/local/lib or ~/waveshare)

Orientation: PORTRAIT (280 wide x 480 tall)

Refresh strategy:
  Splash / boot   → 4-gray full refresh         — branded startup with vine
  Ready / idle    → 1-bit full refresh           — establishes clean baseline
  Scanning        → 1-bit partial refresh        — fast feedback while inference runs
  ID result       → 1-bit partial refresh        — template stays, content updates
  Abstention      → 1-bit partial refresh        — SAME template as ID result

The ID result and abstention screens share an identical template layout so that
partial refresh never ghosts between states. Only the content within each zone
changes — the structural elements (header bar, separator, banner bar) stay put.

Template layout (portrait 280x480):
  +---------------------------+
  | FORAGER          [DOMAIN] |  header  (28px)
  +---------------------------+
  |                           |
  |      [illustration]       |  image zone (200px)
  |       or message          |
  |                           |
  +===========================+  separator line
  |                           |
  |  Species Name             |
  |  Scientific name          |  text zone (~200px)
  |  Confidence: 91% [=====] |
  |  ───────────────────────  |
  |  Lookalike: ...           |
  |  Key diff wrapped text    |
  |                           |
  +---------------------------+
  |       SAFE TO EAT         |  banner (44px)
  +---------------------------+

Safety banner (1-bit — black or white only):
  SAFE    -> WHITE bg / BLACK fg  (open, calm)
  CAUTION -> BLACK bg / WHITE fg  (stands out)
  DEADLY  -> BLACK bg / WHITE fg  (maximum contrast)
  UNKNOWN -> BLACK bg / WHITE fg
"""

import math
import os

from PIL import Image, ImageDraw, ImageFont

from .convergence import ForagerResult

# ── Display constants (PORTRAIT) ──────────────────────────────────────────────
WIDTH  = 280
HEIGHT = 480

BLACK = 0
WHITE = 255

# 4-gray values — used only for splash screen
DARK_GRAY  = 85
LIGHT_GRAY = 170

# Template layout zones
HEADER_H   = 28
ILLUS_TOP  = HEADER_H + 4           # 32
ILLUS_H    = 200
ILLUS_BOT  = ILLUS_TOP + ILLUS_H    # 232
SEP_Y      = ILLUS_BOT + 2          # 234
TEXT_TOP   = SEP_Y + 6              # 240
BANNER_H   = 44
BANNER_TOP = HEIGHT - BANNER_H      # 436
TEXT_BOT   = BANNER_TOP - 4         # 432
X_PAD      = 12
TEXT_W     = WIDTH - X_PAD * 2      # 256

ILLUSTRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "illustrations")

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

# Catch-all classes get cleaned-up display names
DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "high_value_toxics": "Toxic Lookalike",
    "other_mushroom":    "Unknown Mushroom",
}


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


def _wrap_text(draw: ImageDraw.Draw, text: str, font, max_px: int) -> list[str]:
    """Word-wrap text to fit within max_px pixels. Returns list of lines."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textlength(test, font=font) <= max_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


# ── Illustration loading ──────────────────────────────────────────────────────

def _load_illustration(species_key: str) -> Image.Image | None:
    """
    Load a pre-processed illustration PNG for the species.
    Returns None if not found — caller renders text-only layout.
    Scales to fit the illustration zone while maintaining aspect ratio.
    """
    path = os.path.join(ILLUSTRATIONS_DIR, f"{species_key}.png")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("L")
        # Scale to fit the zone while maintaining aspect ratio
        src_w, src_h = img.size
        zone_w = WIDTH - X_PAD * 2
        zone_h = ILLUS_H - 8  # small padding
        scale = min(zone_w / src_w, zone_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return img
    except Exception:
        return None


# ── Template drawing helpers ──────────────────────────────────────────────────

def _draw_header(draw: ImageDraw.Draw, subtitle: str = ""):
    """Draw the FORAGER header strip."""
    draw.rectangle([(0, 0), (WIDTH, HEADER_H)], fill=BLACK)
    font = _load_font(13, bold=True)
    draw.text((8, 6), "WALKING MAN", font=font, fill=WHITE)
    if subtitle:
        bbox = draw.textbbox((0, 0), subtitle, font=font)
        draw.text((WIDTH - bbox[2] - 8, 6), subtitle, font=font, fill=WHITE)


def _draw_separator(draw: ImageDraw.Draw):
    """Draw the horizontal separator between illustration and text zones."""
    draw.line([(X_PAD, SEP_Y), (WIDTH - X_PAD, SEP_Y)], fill=BLACK, width=1)


def _draw_banner(draw: ImageDraw.Draw, label: str, invert: bool = True):
    """
    Draw the bottom banner. In 1-bit mode we only have black and white.
    invert=True  → black bg, white text (CAUTION/DEADLY/UNKNOWN/TRY AGAIN)
    invert=False → white bg, black text (SAFE)
    """
    bg = BLACK if invert else WHITE
    fg = WHITE if invert else BLACK
    font = _load_font(16, bold=True)

    draw.rectangle([(0, BANNER_TOP), (WIDTH, HEIGHT)], fill=bg)
    # Top border line for white-bg banners
    if not invert:
        draw.line([(0, BANNER_TOP), (WIDTH, BANNER_TOP)], fill=BLACK, width=2)
    bbox = draw.textbbox((0, 0), label, font=font)
    bx = (WIDTH - (bbox[2] - bbox[0])) // 2
    by = BANNER_TOP + (BANNER_H - (bbox[3] - bbox[1])) // 2
    draw.text((bx, by), label, font=font, fill=fg)


def _draw_safety_banner(draw: ImageDraw.Draw, safety: str):
    """Draw the safety banner with appropriate styling for 1-bit mode."""
    label = SAFETY_LABEL.get(safety, "UNKNOWN")
    # SAFE gets white bg (calm), everything else gets black bg (attention)
    _draw_banner(draw, label, invert=(safety != "SAFE"))


# ── Vine drawing ──────────────────────────────────────────────────────────────

def _draw_vine(draw: ImageDraw.Draw, progress: float, fill: int = DARK_GRAY,
               leaf_fill: int = DARK_GRAY, vein_fill: int = LIGHT_GRAY):
    """
    Draw a growing vine from the bottom upward along the right side.
    progress: 0.0 (nothing) to 1.0 (fully grown).
    """
    if progress <= 0:
        return

    stem_x = WIDTH - 50
    stem_bottom = HEIGHT - 14
    stem_top = 30
    total_h = stem_bottom - stem_top

    n_points = 50
    points = []
    for i in range(n_points + 1):
        t = i / n_points
        if t > progress:
            break
        y = stem_bottom - t * total_h
        x = stem_x + math.sin(t * math.pi * 3) * 14
        points.append((x, y))

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=fill, width=2)

    leaf_positions = [0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.95]
    for lp in leaf_positions:
        if lp > progress:
            break
        idx = int(lp * n_points)
        if idx >= len(points):
            break
        px, py = points[idx]

        side = 1 if leaf_positions.index(lp) % 2 == 0 else -1
        leaf_len = 16
        leaf_w = 7
        tip_x = px + side * leaf_len
        tip_y = py - 6
        mid_x = px + side * (leaf_len * 0.5)

        draw.polygon([
            (px, py),
            (mid_x, py - leaf_w),
            (tip_x, tip_y),
            (mid_x, py + leaf_w - 3),
        ], fill=leaf_fill)
        draw.line([(px, py), (tip_x, tip_y)], fill=vein_fill, width=1)

    if progress > 0.85 and len(points) > 2:
        tx, ty = points[-1]
        curl_pts = []
        for a in range(0, 100, 10):
            rad = math.radians(a)
            cx = tx + math.cos(rad) * 7
            cy = ty - math.sin(rad) * 7 - 2
            curl_pts.append((cx, cy))
        for i in range(len(curl_pts) - 1):
            draw.line([curl_pts[i], curl_pts[i + 1]], fill=fill, width=1)


# ── Splash / boot screen (4-gray) ────────────────────────────────────────────

SPLASH_QUOTE = (
    '"It is not the strongest of the\n'
    'species that survive, nor the most\n'
    'intelligent, but the one most\n'
    'responsive to change."'
)
SPLASH_ATTRIBUTION = "— Charles Darwin"


def render_splash(vine_progress: float = 0.0) -> Image.Image:
    """
    Boot splash screen: quote, attribution, brand name, and growing vine.
    Rendered in 4-gray for the rich grayscale palette.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    font_quote = _load_font(12)
    font_attr  = _load_font(11, bold=True)
    font_brand = _load_font(16, bold=True)

    # Quote — upper third, centered
    quote_y = 50
    for line in SPLASH_QUOTE.split("\n"):
        _centered_text(draw, line, font_quote, quote_y, DARK_GRAY)
        quote_y += 20

    # Attribution
    _centered_text(draw, SPLASH_ATTRIBUTION, font_attr, quote_y + 10, BLACK)

    # Vine grows up the right side
    _draw_vine(draw, vine_progress)

    # Brand name at the bottom
    _centered_text(draw, "HOMESTEADER LABS", font_brand, HEIGHT - 40, BLACK)

    return img


# ── Ready / idle screen (1-bit full refresh) ─────────────────────────────────

def render_ready() -> Image.Image:
    """
    Idle screen — device is ready for the next scan.
    Uses 1-bit (black/white only). Establishes a clean baseline.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    _draw_header(draw)

    font_ready = _load_font(28, bold=True)
    font_hint  = _load_font(12)

    _centered_text(draw, "READY", font_ready, 180, BLACK)
    _centered_text(draw, 'Say "scan" to identify', font_hint, 228, BLACK)

    # Small vine sprout — bottom right accent
    _draw_vine(draw, 0.18, fill=BLACK, leaf_fill=BLACK, vein_fill=WHITE)

    return img


# ── Scanning screen (1-bit partial refresh) ──────────────────────────────────

def render_scanning() -> Image.Image:
    """
    Fast 1-bit image shown while inference is running.
    Uses the same template structure as the ID screen to avoid ghosting.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    _draw_header(draw, subtitle="SCANNING")
    _draw_separator(draw)

    font_large = _load_font(24, bold=True)
    font_small = _load_font(12)

    # Illustration zone — scanning message
    illus_center_y = ILLUS_TOP + ILLUS_H // 2
    _centered_text(draw, "SCANNING...", font_large, illus_center_y - 20, BLACK)

    # Text zone — guidance
    y = TEXT_TOP + 10
    _centered_text(draw, 'Hold camera 4-6" from subject', font_small, y, BLACK)
    y += 24

    # Progress bar — filled to suggest active work
    bar_w = TEXT_W - 20
    bar_h = 10
    bar_x = X_PAD + 10
    bar_y = y
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], outline=BLACK, width=1)
    fill_w = int(bar_w * 0.7)
    draw.rectangle([(bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1)], fill=BLACK)

    _draw_banner(draw, "IDENTIFYING...", invert=True)

    return img


# ── ID result / abstention render (1-bit partial refresh) ─────────────────────
#
# Both states use the SAME template layout so partial refresh never ghosts.
# The illustration zone and text zone content change; structural elements stay.

def render(result: ForagerResult) -> Image.Image:
    """
    Unified template render for identifications AND abstentions.

    Abstentions (is_unknown) put message text in the illustration zone and
    guidance in the text zone — same structural layout, no ghosting on
    partial refresh.
    """
    img  = Image.new("L", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(img)

    font_species = _load_font(20, bold=True)
    font_sci     = _load_font(12)
    font_conf    = _load_font(14, bold=True)
    font_detail  = _load_font(11)
    font_msg     = _load_font(20, bold=True)
    font_hint    = _load_font(12)

    # ── Structural template (same for every frame) ────────────────────────────
    _draw_separator(draw)

    if result.is_unknown:
        # ── ABSTENTION — message in illustration zone, guidance in text zone ──
        _draw_header(draw)

        illus_center_y = ILLUS_TOP + ILLUS_H // 2
        if result.domain == "other":
            _centered_text(draw, "NOT A FORAGING", font_msg, illus_center_y - 28, BLACK)
            _centered_text(draw, "TARGET", font_msg, illus_center_y + 2, BLACK)

            y = TEXT_TOP + 16
            _centered_text(draw, "Point camera at a berry,", font_hint, y, BLACK)
            _centered_text(draw, "mushroom, or plant.", font_hint, y + 18, BLACK)
        else:
            _centered_text(draw, "MOVE CLOSER", font_msg, illus_center_y - 12, BLACK)

            y = TEXT_TOP + 8
            _centered_text(draw, f"Detected: {result.domain.upper()}", font_hint, y, BLACK)
            y += 22
            _centered_text(draw, "Confidence too low to identify.", font_hint, y, BLACK)
            y += 24
            _centered_text(draw, 'Hold camera 4-6" from subject', font_hint, y, BLACK)
            _centered_text(draw, "and ensure good lighting.", font_hint, y + 18, BLACK)

        _draw_banner(draw, "TRY AGAIN", invert=True)

    elif result.low_confidence:
        # ── LOW CONFIDENCE — same template, uncertain result ──────────────────
        _draw_header(draw, subtitle=result.domain.upper())

        illus_center_y = ILLUS_TOP + ILLUS_H // 2
        _centered_text(draw, "LOW", font_msg, illus_center_y - 28, BLACK)
        _centered_text(draw, "CONFIDENCE", font_msg, illus_center_y + 2, BLACK)

        y = TEXT_TOP + 16
        _centered_text(draw, "Ensure good lighting and", font_hint, y, BLACK)
        _centered_text(draw, "a clear view of the subject.", font_hint, y + 18, BLACK)

        _draw_banner(draw, "TRY AGAIN", invert=True)

    else:
        # ── COMMITTED IDENTIFICATION ──────────────────────────────────────────
        _draw_header(draw, subtitle=result.domain.upper())

        # Illustration zone — centered in the top area
        illus = _load_illustration(result.species)
        if illus is not None:
            iw, ih = illus.size
            ix = (WIDTH - iw) // 2
            iy = ILLUS_TOP + (ILLUS_H - ih) // 2
            img.paste(illus, (ix, iy))

        # Text zone
        y = TEXT_TOP

        # Species display name
        if result.species in DISPLAY_NAME_OVERRIDES:
            display_name = DISPLAY_NAME_OVERRIDES[result.species]
        else:
            raw_name = result.species.replace("_", " ").title()
            display_name = raw_name.replace(" Toxic", "").replace(" Deadly", "").strip()
        draw.text((X_PAD, y), _truncate(draw, display_name, font_species, TEXT_W),
                  font=font_species, fill=BLACK)
        y += 28

        # Scientific name
        draw.text((X_PAD, y), _truncate(draw, result.scientific_name, font_sci, TEXT_W),
                  font=font_sci, fill=BLACK)
        y += 20

        # Confidence: percentage + visual bar
        conf_pct = int(result.confidence * 100)
        conf_label = f"Confidence: {conf_pct}%"
        draw.text((X_PAD, y), conf_label, font=font_conf, fill=BLACK)
        label_w = int(draw.textlength(conf_label, font=font_conf))
        bar_x = X_PAD + label_w + 10
        bar_y_top = y + 4
        bar_h = 10
        bar_max_w = TEXT_W - label_w - 14
        if bar_max_w > 30:
            draw.rectangle([(bar_x, bar_y_top), (bar_x + bar_max_w, bar_y_top + bar_h)],
                           outline=BLACK, width=1)
            fill_w = int(bar_max_w * result.confidence)
            if fill_w > 0:
                draw.rectangle([(bar_x + 1, bar_y_top + 1),
                                (bar_x + fill_w, bar_y_top + bar_h - 1)],
                               fill=BLACK)
        y += 24

        # Lookalike warning
        if result.lookalike and result.lookalike != "N/A":
            draw.line([(X_PAD, y), (X_PAD + TEXT_W, y)], fill=BLACK, width=1)
            y += 6
            lk = _truncate(draw, f"Lookalike: {result.lookalike}", font_detail, TEXT_W)
            draw.text((X_PAD, y), lk, font=font_detail, fill=BLACK)
            y += 16
            if result.key_diff:
                lines = _wrap_text(draw, result.key_diff, font_detail, TEXT_W)
                max_lines = max(1, (TEXT_BOT - y) // 16)
                for line in lines[:max_lines]:
                    draw.text((X_PAD, y), line, font=font_detail, fill=BLACK)
                    y += 16

        _draw_safety_banner(draw, result.safety)

    return img


# ── Display driver interface ──────────────────────────────────────────────────

# Number of partial refreshes before forcing a full 1-bit refresh to clear
# accumulated ghosting artifacts.
_FULL_REFRESH_INTERVAL = 6


class EinkDisplay:
    """
    Thin wrapper around the Waveshare epd3in7 driver.

    Boot sequence (one-time):
      show_splash(progress) — 4-gray boot screen with growing vine
      show_ready()          — 1-bit full refresh, establishes clean baseline

    Scan loop (after first scan, template stays forever):
      show_scanning()       — overlays "SCANNING..." on the banner,
                              previous result stays visible
      show(result)          — committed ID: full template update
                              abstention: overlays status on the banner
    """

    def __init__(self):
        self._epd = None
        self._mode = None
        self._partial_count = 0
        self._last_img = None   # last portrait-orientation rendered image

    def __enter__(self):
        from waveshare_epd import epd3in7
        self._epd = epd3in7.EPD()
        self._set_mode(4)
        self._epd.Clear(WHITE, 0)
        return self

    def __exit__(self, *_):
        if self._epd:
            self._epd.sleep()

    def show_splash(self, vine_progress: float = 0.0):
        """4-gray boot splash with vine animation. Call with increasing progress."""
        img = self._to_driver(render_splash(vine_progress))
        self._set_mode(4)
        self._epd.display_4Gray(self._epd.getbuffer_4Gray(img))

    def show_ready(self):
        """
        1-bit full refresh — establishes a clean baseline.
        Shown once after boot, never again after first scan.
        """
        img = self._to_driver(render_ready())
        self._full_refresh_1bit(img)

    def show_scanning(self):
        """
        If a previous result is on screen, overlay "SCANNING..." on the
        banner — previous ID stays visible, minimal pixel change.
        If no previous result (first scan), full refresh to clear the
        splash/ready screen, then show the scanning template.
        """
        if self._last_img is not None:
            self._show_banner_overlay("SCANNING...")
        else:
            self._full_refresh_1bit(self._to_driver(render_scanning()))

    def show(self, result: ForagerResult):
        """
        Committed identification → full template update, stored as last image.
        Abstention / low confidence → overlay status on the banner,
        previous result stays visible.
        """
        if (result.is_unknown or result.low_confidence) and self._last_img is not None:
            # Transient state — just swap the banner on previous result
            if result.is_unknown and result.domain == "other":
                self._show_banner_overlay("NOT A TARGET")
            elif result.is_unknown:
                self._show_banner_overlay("MOVE CLOSER")
            else:
                self._show_banner_overlay("LOW CONFIDENCE")
        else:
            # Committed result (or first-ever scan) — full template update
            img = render(result)
            self._last_img = img.copy()
            self._partial_refresh(self._to_driver(img))

    def clear(self):
        if self._epd:
            self._set_mode(4)
            self._epd.Clear(WHITE, 0)

    # ── Private ───────────────────────────────────────────────────────────────

    def _show_banner_overlay(self, label: str):
        """Redraw just the banner on the stored last image and partial refresh."""
        img = self._last_img.copy()
        draw = ImageDraw.Draw(img)
        _draw_banner(draw, label, invert=True)
        self._partial_refresh(self._to_driver(img))

    @staticmethod
    def _to_driver(img: Image.Image) -> Image.Image:
        """
        Rotate portrait image (280x480) to the driver's native landscape
        (480x280). The physical portrait mounting handles the rest.
        If the image appears upside-down, change 90 → 270.
        """
        return img.rotate(90, expand=True)

    def _partial_refresh(self, img: Image.Image):
        """
        1-bit partial refresh. Periodically does a full 1-bit refresh
        to clear accumulated ghosting.
        """
        self._set_mode(1)
        if self._partial_count >= _FULL_REFRESH_INTERVAL:
            self._epd.Clear(WHITE, 1)
            self._epd.display_1Gray(self._epd.getbuffer(img))
            self._partial_count = 0
        else:
            self._epd.display_1Gray(self._epd.getbuffer(img))
            self._partial_count += 1

    def _full_refresh_1bit(self, img: Image.Image):
        """Full 1-bit refresh — clears ghosting, resets counter."""
        self._set_mode(1)
        self._epd.Clear(WHITE, 1)
        self._epd.display_1Gray(self._epd.getbuffer(img))
        self._partial_count = 0

    def _set_mode(self, mode: int):
        """Switch between 1-bit (mode=1) and 4-gray (mode=0) only when needed."""
        epd_mode = 0 if mode == 4 else 1
        if self._mode != mode:
            self._epd.init(epd_mode)
            self._mode = mode
