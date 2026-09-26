"""
normalize.py — text cleanup shared by blocking.py and features.py.

Owner: Person 1 (Data Exploration + Normalization)

Everything else in the pipeline depends on this file, so build and
test it FIRST, then hand off normalized columns to the other three.
"""
import re
import unicodedata

# Common business-suffix / word abbreviations seen in this dataset.
# Add to this dict as you spot more patterns during EDA — this is the
# single highest-leverage file in the whole pipeline.
NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "llp": "limited liability partnership",
    "llc": "limited liability company",
    "&": "and",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "apt": "apartment",
    "no": "number",
    "kh": "khasra",  # seen in Indian land-record addresses
}

_WS_RE = re.compile(r"\s+")


def _strip_punct_unicode_safe(text: str) -> str:
    """Remove punctuation while keeping ALL letters and marks, including
    combining vowel signs used in Devanagari and other Indic scripts.

    Python's built-in \\w regex only counts Unicode category 'L' (letters)
    as word characters, NOT category 'M' (combining marks) — so a naive
    [^\\w\\s] regex silently deletes the vowel signs from Devanagari text
    and shreds names like राम मार्केटिंग into unreadable fragments. We
    keep any character whose Unicode category starts with L, M, or N
    (letters, marks, numbers).
    """
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N") or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def basic_clean(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace.
    Safe for both Latin and non-Latin scripts (e.g. Devanagari names) —
    keeps combining marks intact instead of shredding multilingual names.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = _strip_punct_unicode_safe(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def expand_abbreviations(tokens, abbrev_map):
    return [abbrev_map.get(tok, tok) for tok in tokens]


def normalize_business_name(name: str) -> str:
    cleaned = basic_clean(name)
    tokens = cleaned.split()
    tokens = expand_abbreviations(tokens, NAME_ABBREVIATIONS)
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    cleaned = basic_clean(address)
    tokens = cleaned.split()
    tokens = expand_abbreviations(tokens, ADDRESS_ABBREVIATIONS)
    return " ".join(tokens)


def significant_tokens(normalized_name: str, min_len: int = 3):
    """Tokens used as blocking keys — drop very short tokens (they create
    huge, useless blocks) and generic legal-entity words.
    """
    stop = {"private", "limited", "corporation", "company", "incorporated",
            "and", "the", "of"}
    return [t for t in normalized_name.split() if len(t) >= min_len and t not in stop]


if __name__ == "__main__":
    # Quick smoke test — run `python3 src/normalize.py` to sanity check
    samples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "-- Holloway Peak Inc Seafood",
        "Summit Inc",
        "Summit Corporation",
    ]
    for s in samples:
        print(f"{s!r:50s} -> {normalize_business_name(s)!r}")
