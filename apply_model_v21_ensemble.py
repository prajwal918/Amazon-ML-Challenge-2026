#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- V21 MEGA ENSEMBLE  —  BATCH INFERENCE
================================================================================
Applies model_finetuned_v21.pkl to generate submission files.

Bundle keys expected:
  lgb_model, xgb_model (may be None), feature_names, base_threshold,
  sibling_threshold, macro_f05, lgb_weight, xgb_weight

Outputs:
  output/raw_scores_v21.tsv              – raw per-query scores
  output/threshold_sweep_v21/submission_thXX.tsv  – sweep files
  output/matching_results.tsv            – best-threshold submission
  output/matching_results_v21.tsv        – same, versioned copy
================================================================================
"""

import sys, os, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode as _unidecode

# ── XGBoost (optional) ────────────────────────────────────────────────────────
try:
    import xgboost as xgb  # noqa: F401 — imported so pickled model can load
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False
    print("WARNING: xgboost not installed — XGB branch will be skipped")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Metaphone (via jellyfish, optional) ───────────────────────────────────────
_HAVE_METAPHONE = False
try:
    import jellyfish
    _HAVE_METAPHONE = True
except ImportError:
    pass

# ── Indic transliteration (optional) ─────────────────────────────────────────
_HAVE_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _SCRIPT_RANGES = [
        (0x0900, 0x097F, sanscript.DEVANAGARI),
        (0x0B80, 0x0BFF, sanscript.TAMIL),
        (0x0980, 0x09FF, sanscript.BENGALI),
        (0x0C00, 0x0C7F, sanscript.TELUGU),
        (0x0C80, 0x0CFF, sanscript.KANNADA),
    ]
    _HAVE_INDIC = True
except Exception:
    pass

# =============================================================================
#  HELPER FUNCTIONS  (exact copies from finetune_v21_ensemble.py)
# =============================================================================

def metaphone_sim(a, b):
    if not _HAVE_METAPHONE or not a or not b:
        return 0.5
    try:
        ma = jellyfish.metaphone(a)
        mb = jellyfish.metaphone(b)
        if not ma or not mb:
            return 0.5
        return 1.0 if ma == mb else fuzz.ratio(ma, mb) / 100.0
    except:
        return 0.5


def _token_script(token):
    for ch in token:
        code = ord(ch)
        for lo, hi, scheme in _SCRIPT_RANGES:
            if lo <= code <= hi:
                return scheme
    return None


def romanize_mixed(text):
    if not text or not _HAVE_INDIC:
        return text
    out = []
    for tok in text.split():
        scheme = _token_script(tok)
        if scheme:
            try:
                out.append(_translit(tok, scheme, sanscript.ITRANS))
            except:
                out.append(tok)
        else:
            out.append(tok)
    return " ".join(out)


def soundex(word):
    if not word:
        return ""
    word = word.upper()
    first = word[0]
    coded = word.translate(str.maketrans("AEHIOUWY", "00000000"))
    coded = coded.translate(str.maketrans("BFPV", "1111"))
    coded = coded.translate(str.maketrans("CGJKQSXZ", "22222222"))
    coded = coded.translate(str.maketrans("DT", "33"))
    coded = coded.translate(str.maketrans("L", "4"))
    coded = coded.translate(str.maketrans("MN", "55"))
    coded = coded.translate(str.maketrans("R", "6"))
    result = first
    prev = coded[0] if coded else "0"
    for c in coded[1:]:
        if c != prev and c != "0":
            result += c
        prev = c
    return (result + "0000")[:4]


# ── Normalization constants ───────────────────────────────────────────────────
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "llc", "l.l.c",
    "incorporated", "inc", "corp", "corporation", "services", "center",
    "centre", "holdings", "group", "enterprises", "solutions", "llp",
    "l.l.p", "sarl", "sas", "ltd", "limited", "co", "company", "pllc",
    "plc", "gmbh", "sasu", "eurl", "sa", "praiveta limiteda",
    "praivett limitted", "industries", "associates", "consulting",
    "international", "intl",
]
_SUFFIX_RE   = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.IGNORECASE)
_HANDLE_RE   = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE   = re.compile(r"\.(com|net|org|io|co|in|fr|biz|org\.in|gov\.in)\b", re.IGNORECASE)
_DBA_RE      = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*", re.IGNORECASE)
_ABBR_RE     = re.compile(r"^[A-Z]{2,6}$")
_LEET_MAP    = str.maketrans({"5": "s", "3": "e", "1": "i", "0": "o", "4": "a", "7": "t", "8": "b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE       = re.compile(r"\s+")
_STOPWORDS   = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no"}
_NUM_WORD    = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ORDINAL     = {
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
    "first": "1", "second": "2", "third": "3", "fourth": "4",
}

def _clean_basic(s):
    return _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()

def is_abbreviation(s):
    """Check if string looks like an abbreviation (IBM, KFC, etc)"""
    words = s.upper().split()
    return len(words) == 1 and bool(_ABBR_RE.match(words[0])) if words else False

def abbrev_match(q_full, t_full, q_core, t_core):
    """1.0 if one looks like abbreviation of the other"""
    q_initials = "".join(w[0] for w in q_core.split() if w not in _STOPWORDS and len(w) > 1)
    t_initials = "".join(w[0] for w in t_core.split() if w not in _STOPWORDS and len(w) > 1)
    qc, tc = q_core.replace(" ", ""), t_core.replace(" ", "")
    if (q_initials and t_initials and q_initials == t_initials
            and abs(len(q_core.split()) - len(t_core.split())) >= 2):
        return 1.0
    if is_abbreviation(q_core) and t_initials.startswith(q_core[:len(q_core)]):
        return 0.9
    if is_abbreviation(t_core) and q_initials.startswith(t_core[:len(t_core)]):
        return 0.9
    return 0.0

def normalize_name(raw):
    if not isinstance(raw, str) or not raw.strip():
        return {
            "full": "", "core": "", "comp": "", "leet": "", "dba_alt": "",
            "tokens": frozenset(), "qgrams": frozenset(), "initials": "",
            "soundex_toks": frozenset(), "num_tokens": frozenset(),
        }
    s = raw.strip()
    dba_alt = ""
    m = _DBA_RE.search(s)
    if m:
        after = s[m.end():].strip()
        if after:
            dba_alt = _clean_basic(_unidecode(after).lower())
            dba_alt = _WS_RE.sub(" ", _SUFFIX_RE.sub(" ", dba_alt)).strip()
        s = s[:m.start()].strip()
    s = _METADATA_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _DOMAIN_RE.sub(" ", s)
    s = romanize_mixed(s)
    s = _unidecode(s).lower()
    for word, digit in _NUM_WORD.items():
        s = re.sub(r'\b' + word + r'\b', digit, s)
    for word, digit in _ORDINAL.items():
        s = re.sub(r'\b' + word + r'\b', digit, s)
    full = _clean_basic(s)
    core = _WS_RE.sub(" ", _SUFFIX_RE.sub(" ", full)).strip()
    comp = re.sub(r'[^a-z0-9]', '', core)
    leet = ""
    if any(ch.isdigit() for ch in core):
        cand = core.translate(_LEET_MAP)
        if cand != core:
            leet = cand
    tokens      = frozenset(t for t in core.split() if t not in _STOPWORDS and len(t) > 1)
    qgrams      = frozenset(core[i:i+3] for i in range(len(core) - 2)) if len(core) >= 3 else frozenset()
    initials    = "".join(t[0] for t in core.split() if t not in _STOPWORDS and len(t) > 1)
    soundex_toks = frozenset(soundex(t) for t in core.split() if len(t) >= 2)
    num_tokens  = frozenset(t for t in core.split() if t.isdigit())
    return {
        "full": full, "core": core, "comp": comp, "leet": leet, "dba_alt": dba_alt,
        "tokens": tokens, "qgrams": qgrams, "initials": initials,
        "soundex_toks": soundex_toks, "num_tokens": num_tokens,
    }


US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "dl": "delhi",
}
STATE_ABBR    = {**US_STATES, **IN_STATES}
FULL_TO_ABBR  = {v: k for k, v in STATE_ABBR.items()}
FULL_NAME_CANON = {full: full for full in FULL_TO_ABBR}
FULL_NAME_CANON.update({
    "orissa": "odisha", "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry", "tamilnadu": "tamil nadu",
})
_NUM_RE    = re.compile(r"\d+")
_POSTAL_US = re.compile(r"\b\d{5}\b")
_POSTAL_IN = re.compile(r"\b[1-9]\d{5}\b")


def normalize_address(raw, country=""):
    if not isinstance(raw, str) or not raw.strip():
        return {
            "house_number": "", "state": "", "postal": "",
            "tokens": frozenset(), "full": "", "city": "", "is_empty": True,
        }
    s = romanize_mixed(raw)
    s = _unidecode(s).lower()
    s = _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()
    tokens = s.split()
    postal = ""
    if country == "India":
        m = _POSTAL_IN.search(s)
        if m:
            postal = m.group(0)
    else:
        m = _POSTAL_US.search(s)
        if m:
            postal = m.group(0)
    house_number = ""
    for tok in tokens:
        if tok.isdigit() and tok != postal and len(tok) <= 5:
            house_number = tok.lstrip("0") or "0"
            break
    if not house_number:
        for tok in tokens:
            mm = _NUM_RE.search(tok)
            if mm:
                val = mm.group(0)
                if val != postal and len(val) <= 5:
                    house_number = val.lstrip("0") or "0"
                    break
    state = ""
    cleaned_s = re.sub(r'\bfl\b\.?\s*(?:floor|ground|\d)', ' ', s)
    s_tokens  = cleaned_s.split()
    for tok in s_tokens:
        if tok in STATE_ABBR:
            state = STATE_ABBR[tok]
            break
    if not state:
        joined = " " + s + " "
        for full, canon in FULL_NAME_CANON.items():
            if (" " + full + " ") in joined:
                state = canon
                break
    city = ""
    for tok in reversed(tokens):
        if (len(tok) >= 3 and tok not in _STOPWORDS
                and tok not in STATE_ABBR and not tok.isdigit() and tok != postal):
            city = tok
            break
    tok_set = frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
    return {
        "house_number": house_number, "state": state, "postal": postal,
        "tokens": tok_set, "full": s, "city": city, "is_empty": False,
    }


def compute_pair_features(q_norm, q_addr, t_norm, t_addr):
    """Compute the 41-feature vector for a (query, target) pair."""
    q_full, t_full = q_norm["full"], t_norm["full"]
    q_core, t_core = q_norm["core"], t_norm["core"]

    tsort   = fuzz.token_sort_ratio(q_full, t_full) / 100.0
    tset    = fuzz.token_set_ratio(q_full, t_full) / 100.0
    partial = fuzz.partial_ratio(q_full, t_full) / 100.0
    jw      = float(JaroWinkler.similarity(q_full, t_full))
    core_jw   = float(JaroWinkler.similarity(q_core, t_core)) if q_core and t_core else 0.0
    core_sort = fuzz.token_sort_ratio(q_core, t_core) / 100.0 if q_core and t_core else 0.0
    core_set  = fuzz.token_set_ratio(q_core, t_core) / 100.0 if q_core and t_core else 0.0
    exact_core = 1.0 if (q_core and t_core and q_core == t_core) else 0.0
    exact_comp = 1.0 if (q_norm["comp"] and t_norm["comp"] and q_norm["comp"] == t_norm["comp"]) else 0.0

    qlen, tlen = len(q_full), len(t_full)
    len_ratio  = (min(qlen, tlen) / max(qlen, tlen)) if max(qlen, tlen) > 0 else 0.0
    len_diff   = abs(qlen - tlen)

    inter_tok  = len(q_norm["tokens"] & t_norm["tokens"])
    union_tok  = len(q_norm["tokens"] | t_norm["tokens"])
    tok_jaccard = inter_tok / union_tok if union_tok > 0 else 0.0

    inter_qg   = len(q_norm["qgrams"] & t_norm["qgrams"])
    union_qg   = len(q_norm["qgrams"] | t_norm["qgrams"])
    qgram_jacc  = inter_qg / union_qg if union_qg > 0 else 0.0

    used_alt    = 1.0 if (q_norm["dba_alt"] or t_norm["dba_alt"]) else 0.0

    inter_sd   = len(q_norm["soundex_toks"] & t_norm["soundex_toks"])
    union_sd   = len(q_norm["soundex_toks"] | t_norm["soundex_toks"])
    soundex_jacc = inter_sd / union_sd if union_sd > 0 else 0.0

    initials_match = 1.0 if (q_norm["initials"] and t_norm["initials"]
                              and q_norm["initials"] == t_norm["initials"]) else 0.0
    partial_core   = fuzz.partial_ratio(q_core, t_core) / 100.0 if q_core and t_core else 0.0
    leet_match     = 1.0 if (q_norm["leet"] and t_norm["leet"]
                              and q_norm["leet"] == t_norm["leet"]) else 0.0
    qc, tc = q_norm["comp"], t_norm["comp"]
    comp_prefix = 1.0 if (qc and tc and len(qc) >= 4 and len(tc) >= 4 and qc[:8] == tc[:8]) else 0.0
    abbr_match  = abbrev_match(q_full, t_full, q_core, t_core)

    # NEW features
    lev_ratio = Levenshtein.normalized_similarity(q_full, t_full) if q_full and t_full else 0.0
    lev_core  = Levenshtein.normalized_similarity(q_core, t_core) if q_core and t_core else 0.0
    num_match = (
        1.0 if (q_norm["num_tokens"] and t_norm["num_tokens"]
                and q_norm["num_tokens"] == t_norm["num_tokens"])
        else (0.5 if not q_norm["num_tokens"] and not t_norm["num_tokens"] else 0.0)
    )
    meta_sim  = metaphone_sim(
        q_core.split()[0] if q_core else "",
        t_core.split()[0] if t_core else "",
    )
    comp_len_ratio = (
        (min(len(qc), len(tc)) / max(len(qc), len(tc)))
        if qc and tc and max(len(qc), len(tc)) > 0 else 0.0
    )

    # Address features
    if t_addr["is_empty"] or q_addr["is_empty"]:
        house_match = house_conflict = state_match = state_conflict = 0.5
        postal_match = postal_conflict = addr_sort = addr_set = addr_jacc = 0.5
        target_empty = 1.0 if t_addr["is_empty"] else 0.0
        city_match = 0.5
        addr_exact = 0.0
    else:
        target_empty = 0.0
        qh, th = q_addr["house_number"], t_addr["house_number"]
        if qh and th:
            if qh == th:
                house_match, house_conflict = 1.0, 0.0
            elif qh.startswith(th) or th.startswith(qh):
                house_match, house_conflict = 0.8, 0.0
            else:
                house_match, house_conflict = 0.0, 1.0
        else:
            house_match, house_conflict = 0.5, 0.0

        qs, ts = q_addr["state"], t_addr["state"]
        if qs and ts:
            if qs == ts:
                state_match, state_conflict = 1.0, 0.0
            else:
                state_match, state_conflict = 0.0, 1.0
        else:
            state_match, state_conflict = 0.5, 0.0

        qp, tp = q_addr["postal"], t_addr["postal"]
        if qp and tp:
            if qp == tp:
                postal_match, postal_conflict = 1.0, 0.0
            else:
                postal_match, postal_conflict = 0.0, 1.0
        else:
            postal_match, postal_conflict = 0.5, 0.0

        addr_sort = fuzz.token_sort_ratio(q_addr["full"], t_addr["full"]) / 100.0
        addr_set  = fuzz.token_set_ratio(q_addr["full"], t_addr["full"]) / 100.0
        ainter    = len(q_addr["tokens"] & t_addr["tokens"])
        aunion    = len(q_addr["tokens"] | t_addr["tokens"])
        addr_jacc = ainter / aunion if aunion > 0 else 0.0
        qct, tct  = q_addr["city"], t_addr["city"]
        city_match = (1.0 if (qct and tct and qct == tct)
                      else (0.5 if (not qct or not tct) else 0.0))
        addr_exact = (1.0 if (q_addr["full"] and t_addr["full"]
                               and q_addr["full"] == t_addr["full"]) else 0.0)

    name_x_addr = tsort * addr_sort
    core_x_addr  = core_set * addr_set
    jw_x_state   = jw * state_match

    return [
        tsort, tset, partial, jw, core_jw, core_sort, core_set,
        exact_core, exact_comp,
        len_ratio, float(len_diff),
        tok_jaccard, float(inter_tok), qgram_jacc,
        used_alt, soundex_jacc, initials_match, partial_core,
        leet_match, comp_prefix, abbr_match,
        float(lev_ratio), float(lev_core), num_match, float(meta_sim),
        comp_len_ratio,
        house_match, house_conflict, state_match, state_conflict,
        postal_match, postal_conflict, addr_sort, addr_set, addr_jacc,
        city_match, target_empty, addr_exact,
        name_x_addr, core_x_addr, jw_x_state,
    ]


# =============================================================================
#  I/O HELPERS
# =============================================================================

def load_tsv(path, skip_header=True):
    """Yield split rows from a TSV file."""
    with open(path, "r", encoding="utf-8") as fh:
        if skip_header:
            next(fh, None)
        for line in fh:
            yield line.rstrip("\r\n").split("\t")


def read_records(path):
    """Return {id: (name, address, country)} for a source TSV."""
    records = {}
    for parts in load_tsv(path):
        if not parts or not parts[0].strip():
            continue
        eid  = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        addr = parts[2].strip() if len(parts) > 2 else ""
        ctry = parts[3].strip() if len(parts) > 3 else ""
        records[eid] = (name, addr, ctry)
    return records


def read_queries(path):
    """Return {qid: (name, address, country)} from source1 TSV."""
    return read_records(path)


def read_candidate_pairs(path):
    """Return {qid: [tid, ...]} from candidate_pairs.tsv.

    Expected format (with or without header):
        qid<TAB>tid1,tid2,...
    """
    pairs = defaultdict(list)
    first = True
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if first and (parts[0].lower() in ("qid", "query_id", "source1_id")):
                first = False
                continue
            first = False
            if len(parts) < 2:
                continue
            qid  = parts[0].strip()
            tids = [t.strip() for t in parts[1].split(",") if t.strip()]
            pairs[qid] = tids
    return pairs


def write_submission(path, results):
    """Write matching_results style TSV: qid<TAB>tid1,tid2,..."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("query_id\ttarget_ids\n")
        for qid, tids in results:
            fh.write(f"{qid}\t{','.join(tids) if tids else ''}\n")


# =============================================================================
#  CORE INFERENCE
# =============================================================================

def score_queries(
    query_ids,
    query_norm,
    cand_map,
    target_norm,
    lgb_model,
    xgb_model,
    lgb_weight,
    xgb_weight,
    feature_names,
    chunk_size=50_000,
):
    """
    Streaming batch inference.

    Yields (qid, [(tid, prob), ...]) for every query in query_ids.
    Processing is chunked: we accumulate up to `chunk_size` pairs, score
    them in one model call, then yield results for that chunk.
    """
    n_features = len(feature_names)
    buffer_qids  = []   # list of (qid, n_candidates)
    buffer_tids  = []   # flat list of tids for buffer
    buffer_X     = []   # flat list of feature vectors

    total_scored = 0
    t_start = time.time()
    t_chunk  = time.time()

    def _flush(buf_qids, buf_tids, buf_X):
        """Score one chunk and yield per-query results."""
        if not buf_X:
            return
        X = np.array(buf_X, dtype=np.float32)
        lgb_probs = lgb_model.predict_proba(X)[:, 1]
        if xgb_model is not None:
            xgb_probs = xgb_model.predict_proba(X)[:, 1]
            probs = lgb_weight * lgb_probs + xgb_weight * xgb_probs
        else:
            probs = lgb_probs

        offset = 0
        results = []
        for qid, n_cands in buf_qids:
            q_probs = probs[offset: offset + n_cands]
            q_tids  = buf_tids[offset: offset + n_cands]
            scored  = list(zip(q_tids, q_probs.tolist()))
            results.append((qid, scored))
            offset += n_cands
        return results

    chunks_done = 0
    for qid in query_ids:
        q_norm, q_addr = query_norm.get(qid, (None, None))
        if q_norm is None:
            # No normalised record — emit empty result
            yield qid, []
            continue

        cands = cand_map.get(qid, [])
        valid_cands = [(tid, *target_norm[tid]) for tid in cands if tid in target_norm]
        n_cands = len(valid_cands)

        if n_cands == 0:
            yield qid, []
            continue

        # Feature extraction
        feats = [compute_pair_features(q_norm, q_addr, t_norm, t_addr)
                 for tid, t_norm, t_addr in valid_cands]

        buffer_qids.append((qid, n_cands))
        buffer_tids.extend(tid for tid, *_ in valid_cands)
        buffer_X.extend(feats)
        total_scored += n_cands

        # Flush when chunk is full
        if len(buffer_X) >= chunk_size:
            for result in _flush(buffer_qids, buffer_tids, buffer_X):
                yield result
            elapsed = time.time() - t_chunk
            rate    = len(buffer_X) / max(elapsed, 1e-6)
            chunks_done += 1
            print(f"  [chunk {chunks_done}] {total_scored:,} pairs scored | "
                  f"{rate:,.0f} pairs/s | elapsed {time.time()-t_start:.1f}s")
            buffer_qids, buffer_tids, buffer_X = [], [], []
            t_chunk = time.time()

    # Flush remainder
    if buffer_X:
        for result in _flush(buffer_qids, buffer_tids, buffer_X):
            yield result
        chunks_done += 1
        elapsed = time.time() - t_chunk
        rate    = len(buffer_X) / max(elapsed, 1e-6)
        print(f"  [chunk {chunks_done}] {total_scored:,} pairs scored | "
              f"{rate:,.0f} pairs/s | elapsed {time.time()-t_start:.1f}s")


def apply_threshold(scored_queries, base_threshold, sibling_threshold):
    """
    Threshold application: pick top-scoring target if >= base_threshold,
    add siblings >= sibling_threshold.

    scored_queries: iterable of (qid, [(tid, prob), ...])
    Returns list of (qid, [tid, ...]) sorted by qid.
    """
    results = []
    for qid, scored in scored_queries:
        if not scored:
            results.append((qid, []))
            continue
        sorted_pairs = sorted(scored, key=lambda x: -x[1])
        top_tid, top_prob = sorted_pairs[0]
        if top_prob < base_threshold:
            results.append((qid, []))
            continue
        pred = [top_tid]
        for tid, prob in sorted_pairs[1:]:
            if prob >= sibling_threshold:
                pred.append(tid)
            else:
                break
        results.append((qid, pred))
    return results


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("V21 MEGA ENSEMBLE — BATCH INFERENCE")
    print("=" * 80)

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR      = Path(".")
    OUTPUT_DIR    = BASE_DIR / "output"
    SWEEP_DIR     = OUTPUT_DIR / "threshold_sweep_v21"
    MODEL_PATH    = BASE_DIR / "model_finetuned_v21.pkl"
    CAND_PATH     = OUTPUT_DIR / "candidate_pairs.tsv"
    TEST_DIR      = BASE_DIR / "student_resource" / "dataset" / "test"
    Q_PATH        = TEST_DIR / "test_source1.tsv"
    T2_PATH       = TEST_DIR / "test_source2.tsv"
    T3_PATH       = TEST_DIR / "test_source3.tsv"
    RAW_SCORE_PATH     = OUTPUT_DIR / "raw_scores_v21.tsv"
    RESULTS_PATH       = OUTPUT_DIR / "matching_results.tsv"
    RESULTS_V21_PATH   = OUTPUT_DIR / "matching_results_v21.tsv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = 50_000
    SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # ── Load Model Bundle ────────────────────────────────────────────────────
    print(f"\n[1/5] Loading model bundle: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: model bundle not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)

    lgb_model        = bundle["lgb_model"]
    xgb_model        = bundle.get("xgb_model")          # may be None
    feature_names    = bundle["feature_names"]
    base_threshold   = bundle["base_threshold"]
    sibling_threshold = bundle["sibling_threshold"]
    macro_f05        = bundle.get("macro_f05", float("nan"))
    lgb_weight       = bundle.get("lgb_weight", 1.0)
    xgb_weight       = bundle.get("xgb_weight", 0.0)

    ensemble_desc = (
        f"LGB×{lgb_weight:.2f} + XGB×{xgb_weight:.2f}"
        if xgb_model is not None
        else f"LGB×1.0 (no XGB)"
    )
    print(f"  Ensemble   : {ensemble_desc}")
    print(f"  Features   : {len(feature_names)}")
    print(f"  Val F0.5   : {macro_f05:.4f}")
    print(f"  Base th    : {base_threshold:.2f}")
    print(f"  Sibling th : {sibling_threshold:.2f}")

    if xgb_model is None and _HAVE_XGB:
        print("  NOTE: bundle has no XGB model — running LightGBM only")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading data...")

    # Candidate pairs
    if not CAND_PATH.exists():
        sys.exit(f"ERROR: candidate pairs not found at {CAND_PATH}")
    cand_map = read_candidate_pairs(CAND_PATH)
    print(f"  Candidate pairs loaded: {sum(len(v) for v in cand_map.values()):,} "
          f"pairs across {len(cand_map):,} queries")

    # Queries
    if not Q_PATH.exists():
        sys.exit(f"ERROR: test_source1.tsv not found at {Q_PATH}")
    raw_queries = read_queries(Q_PATH)
    print(f"  Queries loaded        : {len(raw_queries):,}")

    # Targets
    raw_targets = {}
    for tpath in [T2_PATH, T3_PATH]:
        if not tpath.exists():
            print(f"  WARNING: {tpath} not found — skipping")
            continue
        rec = read_records(tpath)
        raw_targets.update(rec)
        print(f"  Targets from {tpath.name}: {len(rec):,}")
    print(f"  Total unique targets  : {len(raw_targets):,}")

    # All query IDs in order (preserve file order for determinism)
    all_query_ids = list(raw_queries.keys())

    # ── Normalize ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Normalizing {len(all_query_ids):,} queries and "
          f"{len(raw_targets):,} targets...")
    t_norm_start = time.time()

    # Only normalise targets that appear in at least one candidate list
    needed_tids = set()
    for tids in cand_map.values():
        needed_tids.update(tids)
    needed_tids &= raw_targets.keys()
    print(f"  Needed targets (in candidates): {len(needed_tids):,}")

    query_norm = {}
    for qid, (name, addr, ctry) in raw_queries.items():
        query_norm[qid] = (normalize_name(name), normalize_address(addr, ctry))

    target_norm = {}
    for tid in needed_tids:
        name, addr, ctry = raw_targets[tid]
        target_norm[tid] = (normalize_name(name), normalize_address(addr, ctry))

    print(f"  Normalization done in {time.time() - t_norm_start:.1f}s")

    # ── Scoring ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] Scoring {len(all_query_ids):,} queries "
          f"(chunk_size={CHUNK_SIZE:,})...")

    # Materialise scored results (list of (qid, [(tid, prob), ...]))
    # so we can do threshold sweep without re-running inference.
    all_scored = []   # [(qid, [(tid, prob), ...])]

    scorer_gen = score_queries(
        query_ids      = all_query_ids,
        query_norm     = query_norm,
        cand_map       = cand_map,
        target_norm    = target_norm,
        lgb_model      = lgb_model,
        xgb_model      = xgb_model,
        lgb_weight     = lgb_weight,
        xgb_weight     = xgb_weight,
        feature_names  = feature_names,
        chunk_size     = CHUNK_SIZE,
    )
    for item in scorer_gen:
        all_scored.append(item)

    print(f"  Scoring complete — {len(all_scored):,} queries processed "
          f"in {time.time()-t0:.1f}s total")

    # ── Save Raw Scores ───────────────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")
    print(f"  Writing raw scores → {RAW_SCORE_PATH}")
    with open(RAW_SCORE_PATH, "w", encoding="utf-8") as fh:
        fh.write("qid\tscores\n")
        for qid, scored in all_scored:
            if scored:
                parts = ";".join(f"{tid}:{prob:.6f}" for tid, prob in
                                  sorted(scored, key=lambda x: -x[1]))
            else:
                parts = ""
            fh.write(f"{qid}\t{parts}\n")

    # ── Threshold Sweep ───────────────────────────────────────────────────────
    print(f"  Running threshold sweep over {SWEEP_THRESHOLDS} ...")
    for th in SWEEP_THRESHOLDS:
        sib_th = max(0.25, th - 0.05)
        results = apply_threshold(all_scored, th, sib_th)
        sweep_path = SWEEP_DIR / f"submission_th{int(th*100):02d}.tsv"
        write_submission(sweep_path, results)
        matched = sum(1 for _, tids in results if tids)
        print(f"    th={th:.2f} sib={sib_th:.2f} → matched {matched:,}/{len(results):,} "
              f"({100*matched/max(1,len(results)):.1f}%) → {sweep_path.name}")

    # ── Best-Threshold Submission ─────────────────────────────────────────────
    print(f"  Applying best threshold (base={base_threshold:.2f}, "
          f"sibling={sibling_threshold:.2f})...")
    best_results = apply_threshold(all_scored, base_threshold, sibling_threshold)

    write_submission(RESULTS_PATH, best_results)
    write_submission(RESULTS_V21_PATH, best_results)
    print(f"  Saved → {RESULTS_PATH}")
    print(f"  Saved → {RESULTS_V21_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_q   = len(best_results)
    matched_q = sum(1 for _, tids in best_results if tids)
    singleton_q = total_q - matched_q
    match_rate  = 100.0 * matched_q / max(1, total_q)
    total_elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total queries    : {total_q:,}")
    print(f"  Matched          : {matched_q:,}  ({match_rate:.2f}%)")
    print(f"  Singletons       : {singleton_q:,}")
    print(f"  Match rate       : {match_rate:.2f}%")
    print(f"  Total elapsed    : {total_elapsed:.1f}s")
    print(f"  Val F0.5 (train) : {macro_f05:.4f}")
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
