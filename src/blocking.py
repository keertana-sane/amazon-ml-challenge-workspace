"""
blocking.py — candidate generation (Stage 1 of the pipeline).

Owner: Person 2 (Blocking / Candidate Generation)

Goal: for every Source 1 entity, produce a SMALL shortlist of Source 2 /
Source 3 records that are plausibly the same business. This determines
your recall ceiling — matching.py can only ever find matches that show
up here, so err on the side of a few extra false candidates rather than
missing a true one.

Approach: multi-key inverted-index blocking with IDF-weighted ranking.
  1. Normalize every name / address (via normalize.py) and turn each
     record into a set of blocking KEYS, all scoped by country:
       - name tokens            "memorial", "society"
       - name token PAIRS       ("memorial", "society")
       - concatenated name      "urbanpacificlaunchpad"  (catches domain-style
                                names like urbanpacificlaunchpad.com)
       - address token pairs    ("3807", "lamar"), ("lamar", "avenue") ...
  2. Build an index {key: [record, ...]} over Source 2 + 3 and drop every
     key whose bucket is bigger than MAX_TOKEN_BUCKET_SIZE — too generic to
     be useful, and the thing that keeps this fast.
  3. For each Source 1 entity, every record sharing >=1 key is a candidate.
     Candidates are scored by the sum of the IDF weights of the keys they
     share (a shared rare key counts far more than a shared common one) and
     the top MAX_CANDIDATES_PER_ENTITY are kept.

Why pairs matter: in the first version (single name tokens only, ranked by
raw shared-token count) recall was 29.1%. diagnose_blocking.py showed:
  - 47% of misses shared name tokens, but ALL of them were over the bucket
    cap ("memorial" 5.7K, "retail" 9K, "partners" 263K) — so "Memorial
    Society" got no candidates at all. The PAIR ("memorial", "society") is
    rare, so it survives the cap.
  - 31% of misses shared a kept token but were truncated: a raw shared-token
    count ties hundreds of candidates at 1, so the top-50 cut was arbitrary.
  - 22% shared no name token at all (domains, Devanagari vs Latin names,
    heavy typos) — but ~80% of those share a rare address token.

Implementation notes (for speed on 12M+ records):
  - Key extraction runs in parallel worker processes, one byte range of the
    file each. Keys are hashed to 64-bit ints with blake2b (stable across
    processes, unlike Python's salted hash()).
  - The index and the Source1 x candidate join are numpy sort / searchsorted
    operations rather than Python dicts of lists.
"""
import csv
import hashlib
import os
import re
import sys
import time
import unicodedata
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_business_name, normalize_address, significant_tokens

MAX_CANDIDATES_PER_ENTITY = 50  # cap to keep matching.py's workload sane

# Some keys (e.g. "liability" from LLC/LLP expansions, generic words like
# "center", "partners", "group", "services", "holdings", or Hindi words for
# "Limited"/"Private") appear in tens or hundreds of THOUSANDS of records.
# A key that common carries almost no information for blocking — it doesn't
# help tell businesses apart — but every entity that contains it would still
# have to score every one of those candidates, which is what was silently
# stalling the pipeline. We drop any key bucket bigger than this threshold.
MAX_TOKEN_BUCKET_SIZE = 2000

N_WORKERS = max(1, min(10, os.cpu_count() or 1))
FANOUT_BUDGET = 40_000_000  # joined (entity, posting) rows per batch — bounds peak memory
ENTITY_FANOUT_BUDGET = 300  # postings expanded per Source 1 entity, rarest keys first — bounds total work

# Key types. Kept as small ints so each posting can carry its type, which
# lets diagnose/tuning code switch key types on and off.
K_NAME_TOKEN, K_NAME_PAIR, K_NAME_CONCAT, K_ADDR_PAIR = 1, 2, 3, 4
ENABLED_KEY_TYPES = (K_NAME_TOKEN, K_NAME_PAIR, K_NAME_CONCAT, K_ADDR_PAIR)

# Relative weight of each key type in the candidate score.
KEY_TYPE_WEIGHT = {K_NAME_TOKEN: 1.0, K_NAME_PAIR: 1.0, K_NAME_CONCAT: 1.0, K_ADDR_PAIR: 1.0}

# Extra words ignored for blocking only (normalize.py's stop list already
# drops private/limited/company/...). These come out of the LLC/LLP/PC
# expansions or are pure web noise, and would otherwise pollute pair and
# concat keys ("pressholdingliability" vs "pressholding").
NAME_BLOCK_STOP = {"liability", "partnership", "llc", "llp", "pc", "lp", "plc",
                   "com", "www", "net", "org", "http", "https",
                   # Private / Limited written in Devanagari
                   "प्राइवेट", "प्रा", "लिमिटेड", "लि"}

# Very common address words: pairs made of two of these are useless, so an
# address token pair must contain at least one token outside this set.
ADDR_GENERIC = {"street", "road", "avenue", "drive", "lane", "boulevard", "court",
                "place", "way", "circle", "highway", "suite", "unit", "floor",
                "number", "near", "null", "none", "door", "h", "flat", "building",
                "apartment", "plot", "opp", "east", "west", "north", "south",
                "n", "s", "e", "w", "the", "of", "and", "st", "rd", "nagar"}

MAX_NAME_TOKENS_FOR_PAIRS = 6
MAX_ADDR_TOKENS = 12

_DIGIT_LOOKALIKE = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "8": "b"})
_ALNUM_MIX = re.compile(r"(?=.*[a-z])(?=.*\d)")
_LEADING_NUM = re.compile(r"^(\d+)[a-z]?$")


def _strip_latin_accents(text):
    """'Déntal' -> 'Dental'. Only strips combining marks that sit on a Latin
    base letter — Devanagari / other Indic vowel signs are left untouched.
    """
    if text.isascii():
        return text
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn" and out and "a" <= out[-1].lower() <= "z":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _fix_name_token(tok):
    """'harb0r' -> 'harbor', 'at1antic' -> 'atlantic': digits used as letter
    look-alikes inside an otherwise alphabetic token."""
    if _ALNUM_MIX.match(tok):
        fixed = tok.translate(_DIGIT_LOOKALIKE)
        if fixed.isalpha():
            return fixed
    return tok


def name_block_tokens(name):
    norm = normalize_business_name(_strip_latin_accents(name or ""))
    toks = [_fix_name_token(t) for t in significant_tokens(norm)]
    seen, out = set(), []
    for t in toks:
        if t not in NAME_BLOCK_STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _addr_token(tok):
    m = _LEADING_NUM.match(tok)
    if m:
        return m.group(1).lstrip("0") or "0"
    return tok


def addr_block_parts(address):
    """Address -> list of token lists, one per comma-separated component.
    Components are kept separate because their ORDER varies wildly across
    sources ("TN, MEMPHIS, 03807 LAMAR AVE" vs "3807 Lamar Avenue, Memphis").
    """
    parts = []
    for part in _strip_latin_accents(address or "").split(","):
        toks = [_addr_token(t) for t in normalize_address(part).split()]
        toks = [t for t in toks if len(t) >= 2 or t.isdigit()]
        if toks:
            parts.append(toks)
    return parts


def record_keys(name, address, country):
    """All blocking keys for one record, as a list of (key_type, key_string)."""
    keys = []
    ntoks = name_block_tokens(name)
    for t in ntoks:
        keys.append((K_NAME_TOKEN, t))
    pt = ntoks[:MAX_NAME_TOKENS_FOR_PAIRS]
    for i in range(len(pt)):
        for j in range(i + 1, len(pt)):
            a, b = (pt[i], pt[j]) if pt[i] < pt[j] else (pt[j], pt[i])
            keys.append((K_NAME_PAIR, a + "|" + b))
    concat = "".join(ntoks)
    if len(concat) >= 6:
        keys.append((K_NAME_CONCAT, concat))

    # Address pairs: every pair of tokens within one comma-separated part,
    # plus (house-number, token) across the whole address. Pairs made only
    # of generic words are skipped.
    parts = addr_block_parts(address)
    seen_pairs = set()
    budget = MAX_ADDR_TOKENS
    numbers = []
    for toks in parts:
        toks = toks[:budget]
        budget -= len(toks)
        for i in range(len(toks)):
            if toks[i].isdigit():
                numbers.append(toks[i])
            for j in range(i + 1, len(toks)):
                a, b = toks[i], toks[j]
                if a == b or (a in ADDR_GENERIC and b in ADDR_GENERIC):
                    continue
                seen_pairs.add((a, b) if a < b else (b, a))
        if budget <= 0:
            break
    if numbers:
        all_alpha = {t for toks in parts for t in toks if not t.isdigit() and t not in ADDR_GENERIC}
        for num in numbers[:2]:
            for t in all_alpha:
                seen_pairs.add((num, t) if num < t else (t, num))
    for a, b in seen_pairs:
        keys.append((K_ADDR_PAIR, a + "|" + b))

    return [(kt, country + "\x1f" + str(kt) + "\x1f" + k) for kt, k in keys]


def _hash64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def _file_ranges(path, n):
    """Split a file into n byte ranges aligned to line starts (header skipped)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.readline()
        start0 = f.tell()
        bounds = [start0]
        for i in range(1, n):
            f.seek(max(start0, size * i // n))
            f.readline()
            bounds.append(max(f.tell(), bounds[-1]))
        bounds.append(size)
    return [(path, bounds[i], bounds[i + 1]) for i in range(n) if bounds[i + 1] > bounds[i]]


def _extract_range(args):
    """Worker: parse one byte range of a source TSV, return its entity ids
    and the (key_hash, local_record_index, key_type) postings."""
    path, start, end = args
    ids = []
    keys, recs, types = [], [], []
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8")
    for row in csv.reader(data.splitlines(), delimiter="\t"):
        if not row:
            continue
        row += [""] * (4 - len(row))
        idx = len(ids)
        ids.append(row[0])
        seen = set()
        for kt, k in record_keys(row[1], row[2], row[3]):
            h = _hash64(k)
            if h not in seen:
                seen.add(h)
                keys.append(h)
                recs.append(idx)
                types.append(kt)
    return (ids, np.array(keys, dtype=np.int64), np.array(recs, dtype=np.int32),
            np.array(types, dtype=np.int8))


def extract_keys(paths, pool):
    """Parallel key extraction over one or more source files. Returns
    (entity_ids list, key_hash array, record_index array, key_type array)."""
    tasks = []
    for p in paths:
        tasks.extend(_file_ranges(p, N_WORKERS * 2))
    ids = []
    keys, recs, types = [], [], []
    for r_ids, r_keys, r_recs, r_types in pool.imap(_extract_range, tasks):
        keys.append(r_keys)
        recs.append(r_recs + len(ids))
        types.append(r_types)
        ids.extend(r_ids)
    return ids, np.concatenate(keys), np.concatenate(recs), np.concatenate(types)


class BlockingIndex:
    """Sorted-array inverted index over Source 2 + 3 postings.

    query_keys (optional): only keys that some Source 1 record actually has
    are kept — a key no query can hit is dead weight. Bucket sizes (and so
    the bucket cap and IDF weights) are computed BEFORE this pruning.
    """

    def __init__(self, keys, recs, types, key_types=ENABLED_KEY_TYPES, query_keys=None):
        if not np.isin(np.unique(types), key_types).all():
            mask = np.isin(types, key_types)
            keys, recs, types = keys[mask], recs[mask], types[mask]
            del mask
        order = np.argsort(keys)
        keys, self.recs, types = keys[order], recs[order].astype(np.int32), types[order]
        del order
        # Bucket boundaries straight from the sorted keys (np.unique would sort again).
        new_key = np.ones(len(keys), dtype=bool)
        new_key[1:] = keys[1:] != keys[:-1]
        start = np.flatnonzero(new_key)
        del new_key
        count = np.diff(np.append(start, len(keys)))
        ukeys = keys[start]
        utype = types[start]  # the key string embeds its type, so any posting will do
        del keys, types
        oversized = count > MAX_TOKEN_BUCKET_SIZE
        self.n_dropped = int(oversized.sum())
        keep = ~oversized
        if query_keys is not None:
            qk = np.unique(query_keys)
            p = np.minimum(np.searchsorted(qk, ukeys), len(qk) - 1)
            keep &= qk[p] == ukeys
            del qk, p
        self.n_keys = len(ukeys)
        self.ukeys, self.start, self.count = ukeys[keep], start[keep], count[keep].astype(np.int32)
        type_w = np.array([KEY_TYPE_WEIGHT.get(t, 1.0) for t in range(max(KEY_TYPE_WEIGHT) + 1)])
        # IDF-style weight: a key shared by 1 record ~7.6, by 2000 records ~0.7.
        self.weight = (np.log1p(MAX_TOKEN_BUCKET_SIZE / self.count) * type_w[utype[keep]]).astype(np.float32)

    def lookup(self, q_keys):
        """Index position of each query key, or -1 when absent / dropped."""
        if len(self.ukeys) == 0:
            return np.full(len(q_keys), -1, np.int64)
        pos = np.minimum(np.searchsorted(self.ukeys, q_keys), len(self.ukeys) - 1)
        pos[self.ukeys[pos] != q_keys] = -1
        return pos

    def candidates(self, pos, owner, max_candidates):
        """pos: index positions of hit query keys (from lookup, -1 removed);
        owner: local Source 1 index of each. Returns (owner, candidate_record,
        score), top-N per owner, sorted by owner then descending score."""
        cnt = self.count[pos].astype(np.int64)
        total = int(cnt.sum())
        self.last_fanout = total
        if total == 0:
            e = np.array([], np.int64)
            return e, e, np.array([], np.float32)
        # Expand every (query key -> bucket) into one row per posting.
        row = np.repeat(np.arange(len(pos), dtype=np.int32), cnt)
        offs = np.arange(total, dtype=np.int64)
        offs += np.repeat(self.start[pos] - (np.cumsum(cnt) - cnt), cnt)
        code = owner[row].astype(np.int64) << 32
        code |= self.recs[offs]
        del offs
        w = self.weight[pos][row]
        del row
        # Sum weights per (owner, candidate).
        order = np.argsort(code)
        code, w = code[order], w[order]
        del order
        first = np.ones(len(code), dtype=bool)
        first[1:] = code[1:] != code[:-1]
        starts = np.flatnonzero(first)
        del first
        score = np.add.reduceat(w, starts)
        code = code[starts]
        del w, starts
        u_owner = code >> 32
        u_cand = code & 0xFFFFFFFF
        # Top-N per owner: sort by owner asc, score desc.
        order = np.lexsort((-score, u_owner))
        u_owner, u_cand, score = u_owner[order], u_cand[order], score[order]
        rank = np.arange(len(u_owner)) - np.searchsorted(u_owner, u_owner, side="left")
        keep = rank < max_candidates
        return u_owner[keep], u_cand[keep], score[keep]


def iter_candidate_batches(index, q_keys, q_recs, n_s1, max_candidates):
    """Yield (lo, hi, owner, cand) over all Source 1 records, in batches
    sized so the joined row count stays under FANOUT_BUDGET (bounds memory).
    q_recs must be sorted."""
    pos = index.lookup(q_keys)
    hit = pos >= 0
    pos, own = pos[hit], q_recs[hit]
    del hit
    # Per-entity budget: expand each entity's keys rarest-first and stop once
    # ENTITY_FANOUT_BUDGET postings have been pulled (the rarest key is always
    # used). Rare shared keys — (house number, street), (name word, name word)
    # — are what true matches have in common; a candidate reachable only via a
    # 1,500-record bucket scores < 1 and would not make the top-N anyway.
    cnt = index.count[pos].astype(np.int64)
    order = np.lexsort((cnt, own))
    pos, own, cnt = pos[order], own[order], cnt[order]
    del order
    cum = np.cumsum(cnt)
    first = np.searchsorted(own, own, side="left")
    before = cum - cnt - np.where(first > 0, cum[first - 1], 0)  # postings used by rarer keys of this entity
    keep = (before < ENTITY_FANOUT_BUDGET) & ((before + cnt <= ENTITY_FANOUT_BUDGET) | (np.arange(len(own)) == first))
    pos, own = pos[keep], own[keep]
    del cnt, cum, first, before, keep
    fan = np.bincount(own, weights=index.count[pos], minlength=n_s1)
    cum = np.cumsum(fan)
    lo = 0
    while lo < n_s1:
        base = cum[lo - 1] if lo else 0.0
        hi = int(np.searchsorted(cum, base + FANOUT_BUDGET, side="right"))
        hi = min(max(hi, lo + 1), n_s1)
        a, b = np.searchsorted(own, [lo, hi])
        owner, cand, _ = index.candidates(pos[a:b], own[a:b] - lo, max_candidates)
        yield lo, hi, owner, cand
        lo = hi


def load_source(path):
    """Load a source file into a list of dicts: entity_id, business_name,
    business_address, country. Reads as UTF-8 TSV — required for the
    Devanagari and other non-Latin names in this dataset.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def get_candidates(source1_record, index, cand_ids, max_candidates=MAX_CANDIDATES_PER_ENTITY):
    """Look up candidates for a single Source 1 record (handy for debugging;
    the batch path in generate_candidate_pairs does the same thing for all
    records at once)."""
    ks = {_hash64(k) for _, k in record_keys(source1_record.get("business_name", ""),
                                             source1_record.get("business_address", ""),
                                             source1_record.get("country", ""))}
    pos = index.lookup(np.array(sorted(ks), dtype=np.int64))
    pos = pos[pos >= 0]
    _, cands, _ = index.candidates(pos, np.zeros(len(pos), np.int64), max_candidates)
    return [cand_ids[c] for c in cands]


def generate_candidate_pairs(source1_path, source2_path, source3_path, output_path,
                             max_candidates=MAX_CANDIDATES_PER_ENTITY,
                             key_types=ENABLED_KEY_TYPES):
    """End-to-end: extract keys from everything, build one combined index over
    Source 2 + Source 3, generate candidates for every Source 1 record, write
    candidate_pairs.tsv in the exact format the challenge requires.
    """
    t0 = time.time()
    with get_context("fork").Pool(N_WORKERS) as pool:
        print("Extracting blocking keys from Source 2 + Source 3...")
        cand_ids, keys, recs, types = extract_keys([source2_path, source3_path], pool)
        print(f"  {len(cand_ids):,} records, {len(keys):,} key postings  ({time.time() - t0:.0f}s)")
        print("Extracting blocking keys from Source 1...")
        s1_ids, q_keys, q_recs, q_types = extract_keys([source1_path], pool)
        print(f"  {len(s1_ids):,} records, {len(q_keys):,} key postings  ({time.time() - t0:.0f}s)")

    qmask = np.isin(q_types, key_types)
    q_keys, q_recs = q_keys[qmask], q_recs[qmask]
    del q_types, qmask

    print("Building inverted index...")
    index = BlockingIndex(keys, recs, types, key_types, query_keys=q_keys)
    del keys, recs, types
    print(f"  {index.n_keys:,} distinct keys; dropped {index.n_dropped:,} overly generic "
          f"(bucket > {MAX_TOKEN_BUCKET_SIZE:,}); {len(index.ukeys):,} usable by Source 1  "
          f"({time.time() - t0:.0f}s)")

    print("Generating candidates for every Source 1 entity...")
    n_pairs = 0
    n_empty = 0
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        # q_recs is already in ascending order: extract_keys emits records in file order.
        for lo, hi, owner, cand in iter_candidate_batches(index, q_keys, q_recs, len(s1_ids), max_candidates):
            bounds = np.searchsorted(owner, np.arange(hi - lo + 1))
            cand = cand.tolist()
            lines = []
            for i in range(hi - lo):
                c = cand[bounds[i]:bounds[i + 1]]
                if not c:
                    n_empty += 1
                lines.append(s1_ids[lo + i] + "\t" + ",".join([cand_ids[x] for x in c]) + "\n")
            n_pairs += len(cand)
            out.writelines(lines)
            if hi == len(s1_ids) or hi // 200_000 != lo // 200_000:
                print(f"  Processed {hi:,} / {len(s1_ids):,} entities  ({time.time() - t0:.0f}s)")

    print(f"Done. Wrote {output_path}: {n_pairs:,} candidate pairs, "
          f"{n_pairs / max(1, len(s1_ids)):.1f} per entity, {n_empty:,} entities with none. "
          f"Total time {time.time() - t0:.0f}s")


def measure_recall(candidate_pairs_path, ground_truth_path):
    """Quick sanity check: of all TRUE matches in ground truth, what
    fraction actually appear in our candidate lists? This is the recall
    CEILING for the whole pipeline — matching.py can never recover a
    true match that isn't here. Run this after every blocking change.
    """
    candidates = {}
    with open(candidate_pairs_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            cand_ids = set(row[1].split(",")) if len(row) > 1 and row[1] else set()
            candidates[s1_id] = cand_ids

    total_true_matches = 0
    found_true_matches = 0
    with open(ground_truth_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            true_matches = set(row[1].split(",")) if len(row) > 1 and row[1] else set()
            total_true_matches += len(true_matches)
            found_true_matches += len(true_matches & candidates.get(s1_id, set()))

    recall = found_true_matches / total_true_matches if total_true_matches else 0.0
    print(f"Recall ceiling: {found_true_matches:,} / {total_true_matches:,} = {recall:.1%}")
    return recall


if __name__ == "__main__":
    # Run this from your repo root:
    #   python3 src/blocking.py          -> train candidates + recall
    #   python3 src/blocking.py test     -> test candidates
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        generate_candidate_pairs(
            source1_path="dataset/test/test_source1.tsv",
            source2_path="dataset/test/test_source2.tsv",
            source3_path="dataset/test/test_source3.tsv",
            output_path="output/candidate_pairs_test.tsv",
        )
    else:
        generate_candidate_pairs(
            source1_path="dataset/train/train_source1.tsv",
            source2_path="dataset/train/train_source2.tsv",
            source3_path="dataset/train/train_source3.tsv",
            output_path="output/candidate_pairs_train.tsv",
        )
        measure_recall("output/candidate_pairs_train.tsv", "dataset/train/train_ground_truth.tsv")
