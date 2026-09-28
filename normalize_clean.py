import re, unicodedata
from anyascii import anyascii

DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz", ".org.in", ".gov.in")
LEGAL_SUFFIXES_SET = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises",
    "industries", "associates", "consulting", "solutions", "international", "intl"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no", "mr", "ms", "sri", "shri", "smt"}

# Leet-speak mapping when mixed inside words
LEET_MAP = {
    '5': 's',
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '@': 'a',
    '$': 's',
    '8': 'b',
    '7': 't',
}

def un_leetspeak_word(word):
    """If word has digits mixed with letters (e.g. '5mart', 'm00re', 'b1twise'), convert leetspeak digits."""
    has_letters = any(c.isalpha() for c in word)
    has_digits = any(c.isdigit() or c in ('@', '$') for c in word)
    
    if has_letters and has_digits:
        # Check if it looks like a year or phone or address number
        if re.fullmatch(r'\d+[a-zA-Z]?', word):  # e.g. 12A, 100
            return word
        chars = [LEET_MAP.get(c, c) for c in word]
        return "".join(chars)
    return word

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def clean_name_pre_vectorize(name):
    """
    Gold standard normalization BEFORE building blocking indices or vectorizing.
    Resolves accents, transliterates non-Latin scripts, de-leetspeaks, strips legal suffixes.
    """
    # 1. Transliterate to Latin ASCII (handles Hindi, Bengali, Arabic, etc.)
    name = anyascii(str(name)).lower().strip()
    
    # 2. Strip accents
    name = strip_accents(name)
    
    # 3. Strip domains
    for ext in DOMAINS:
        if name.endswith(ext):
            name = name[:-len(ext)].strip()
            
    # 4. Handle '&' and '+'
    name = name.replace("&", " and ").replace("+", " and ")
    
    # 5. Tokenize and un-leetspeak
    tokens = re.findall(r'[a-zA-Z0-9@$]+', name)
    cleaned_toks = []
    for tok in tokens:
        tok = un_leetspeak_word(tok)
        if tok not in LEGAL_SUFFIXES_SET and tok not in STOPWORDS and len(tok) > 1:
            cleaned_toks.append(tok)
            
    return " ".join(cleaned_toks)

def clean_addr_pre_vectorize(addr):
    """Normalize address: remove leading zeros, handle accents and non-Latin."""
    addr = anyascii(str(addr)).lower().strip()
    addr = strip_accents(addr)
    # Strip leading zeros from numeric tokens: "0337" -> "337", "##20" -> "20"
    addr = re.sub(r'\b0+(\d+)\b', r'\1', addr)
    addr = re.sub(r'[^\w\s]', ' ', addr)
    return " ".join(addr.split())

if __name__ == "__main__":
    # Test on known noise cases
    test_cases = [
        "5mart Solutions Pvt Ltd",
        "Economic Dvéegpemnt United Fellowship",
        "WBILMIASON NETORK .com",
        "ফার্স্ট ফুড",  # Bengali
        "ऑल इंटरनेशनल", # Hindi
        "00337 Main St ##20",
    ]
    print("Testing Pre-Vectorization Normalization:")
    for t in test_cases:
        print(f"  Raw:   '{t}'")
        print(f"  Clean: '{clean_name_pre_vectorize(t)}'")
        print()
