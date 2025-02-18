from typing import Optional


def split_keywords(keywords_raw: Optional[str]) -> list[str]:
    """Split a raw keywords string into a list of keywords.

    Try splitting on some different characters until we are successful (i.e. splitting produced at least two parts).
    Try first on ";", then on ",", then on any whitespace character.
    """
    keywords_raw = keywords_raw.strip() if keywords_raw else None
    # Ignore one-character keyword strings, like "|"
    if not keywords_raw or len(keywords_raw) <= 1:
        return []

    # Try splitting on ";", then on ",", then on any whitespace character.
    # Splitting must produce at least two parts, otherwise fallback to full keywords_raw.
    return (
        _try_splitting(keywords_raw, ";")
        or _try_splitting(keywords_raw, ",")
        or _try_splitting(keywords_raw, None)
        or [keywords_raw]
    )


def _try_splitting(text: str, sep: str | None) -> list[str] | None:
    # Split on separator, strip each part, remove empty parts.
    # Return only if splitting produced at least two parts.
    parts = text.split(sep)
    parts = [p.strip() for p in parts]
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else None
