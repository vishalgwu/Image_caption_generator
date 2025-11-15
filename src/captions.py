# src/captions.py

import math

def _season_phrase(season: str) -> str:
    """
    Map raw season text to a generic marketing phrase.
    We never mention the year, only general seasonal wording.
    """
    if not isinstance(season, str):
        return "perfect for everyday wear"

    season = season.strip().lower()
    if season in ("fall", "autumn"):
        return "perfect for cooler months"
    if season in ("winter",):
        return "ideal for cold weather and layering"
    if season in ("spring",):
        return "great for mild spring days"
    if season in ("summer",):
        return "perfect for warm, sunny days"

    return "perfect for everyday wear"


def _gender_phrase(gender: str) -> str:
    """
    Turn raw gender into marketing-friendly phrase.
    """
    if not isinstance(gender, str):
        return "everyone"

    g = gender.strip().lower()
    if g == "men":
        return "men"
    if g == "women":
        return "women"
    if g == "boys":
        return "boys"
    if g == "girls":
        return "girls"

    return "everyone"


def build_caption(row) -> str:
    """
    Build a single caption string from a metadata row.
    Expected columns in row:
      - productDisplayName
      - articleType
      - baseColour
      - gender
      - season
      - usage (optional)
    """
    name = str(row.get("productDisplayName", "")).strip()
    article = str(row.get("articleType", "")).strip()
    colour = str(row.get("baseColour", "")).strip()
    gender = _gender_phrase(row.get("gender", ""))
    season_phrase = _season_phrase(row.get("season", ""))
    usage = str(row.get("usage", "")).strip()

    parts = []

    # Start with "A ..."
    if colour and article:
        parts.append(f"A {colour.lower()} {article.lower()}")
    elif article:
        parts.append(f"A {article.lower()}")
    elif name:
        parts.append(name)
    else:
        parts.append("A stylish piece")

    # Add gender phrase
    if gender != "everyone":
        parts[-1] += f" for {gender}"

    # Add usage if available
    if usage:
        parts.append(f"designed for {usage.lower()} use")

    # Add season phrase (generic, no year)
    if season_phrase:
        parts.append(season_phrase)

    # Join into one sentence
    caption = ", ".join(parts)
    # Ensure it ends with a period
    if not caption.endswith("."):
        caption += "."

    return caption
