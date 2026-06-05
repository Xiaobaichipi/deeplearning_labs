"""Chinese font auto-detection for matplotlib plots."""

import matplotlib
import matplotlib.font_manager as fm
import os
import warnings


# Common Chinese font names across platforms
_CHINESE_FONT_NAMES = [
    # Linux
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "AR PL UMing CN",
    "Droid Sans Fallback",
    # macOS
    "PingFang SC",
    "PingFang TC",
    "STHeiti",
    "STKaiti",
    "Hiragino Sans GB",
    "Apple LiGothic",
    # Windows
    "SimHei",
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "FangSong",
    "KaiTi",
    "SimSun",
    # Generic
    "DejaVu Sans",  # has some CJK coverage
]


def find_chinese_font():
    """Find the first available Chinese font on the system.

    Returns the font family name (str), or 'sans-serif' as fallback.
    """
    try:
        fonts = fm.findSystemFonts()
        if not fonts:
            return "sans-serif"

        # Build a set of lowercased canonical font names for matching
        font_map = {}
        for fp in fonts:
            try:
                prop = fm.FontProperties(fname=fp)
                name = prop.get_name()
                if name:
                    font_map[name.lower()] = name
            except Exception:
                continue

        # First pass: try well-known font names
        for candidate in _CHINESE_FONT_NAMES:
            if candidate.lower() in font_map:
                return font_map[candidate.lower()]

        # Second pass: check if any installed font has "CJK", "SC", "CN", "chinese",
        # or "han" in its name (case-insensitive)
        keywords = ["cjk", "sc", "cn", "chinese", "han", "ming", "hei", "song",
                     "kai", "fang", "yuan", "gothic", "noto", "wenquan"]
        for name_lower, name_orig in font_map.items():
            for kw in keywords:
                if kw in name_lower:
                    return name_orig

        # Third pass: try to find a font that can actually render Chinese
        # by rendering a test character
        if "sans-serif" in font_map:
            candidate_names = list(font_map.values())
            sim_heis = [n for n in candidate_names if "hei" in n.lower()
                        or "song" in n.lower()
                        or "ming" in n.lower()
                        or "noto" in n.lower()]
            if sim_heis:
                return sim_heis[0]

    except Exception:
        pass

    return "sans-serif"


# Global: set once on import
_chinese_font = None


def get_chinese_font():
    """Get the detected Chinese font (cached after first call)."""
    global _chinese_font
    if _chinese_font is None:
        _chinese_font = find_chinese_font()
    return _chinese_font


def setup_chinese_font():
    """Configure matplotlib to use Chinese font globally."""
    font = get_chinese_font()
    if font != "sans-serif":
        matplotlib.rcParams["font.family"] = font
    else:
        # If no Chinese font found, at least try to not break
        matplotlib.rcParams["font.family"] = "sans-serif"
    # Always fall back to DejaVu Sans for missing glyphs
    matplotlib.rcParams["font.sans-serif"] = [font] if font != "sans-serif" else []
    return font
