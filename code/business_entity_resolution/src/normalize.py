"""
Normalization and Token Extraction Utilities for Business Entity Resolution.
Handles address abbreviation expansions, legal suffix removal, and number extraction.
"""

import re

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
}

STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by"}

ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "hwy": "highway",
    "ct": "court", "pl": "place", "ste": "suite", "fl": "floor", "pkg": "parking",
}


def normalize_text(text: str) -> str:
    """Lowercase, expand ampersands, strip non-alphanumeric, collapse whitespace."""
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_name_tokens(name: str) -> frozenset:
    """Extract informative business name tokens after legal suffix & stopword filtering."""
    toks = normalize_text(name).split()
    return frozenset(t for t in toks if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1)


def extract_addr_tokens(addr: str) -> frozenset:
    """Extract normalized address tokens with canonical abbreviation mapping."""
    toks = normalize_text(addr).split()
    return frozenset(ADDRESS_ABBREV.get(t, t) for t in toks if t not in STOPWORDS and len(t) > 1)


def extract_numbers(addr: str) -> frozenset:
    """Extract standalone numeric components (building/shop numbers, PIN codes)."""
    return frozenset(re.findall(r"\b\d+\b", str(addr)))
