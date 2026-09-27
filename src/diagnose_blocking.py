"""
diagnose_blocking.py — explain WHY blocking misses true matches.

Owner: Person 2 (Blocking / Candidate Generation)

Run from repo root AFTER blocking.py has written a candidate file:
    python3 src/diagnose_blocking.py [candidate_pairs_path]

What it reports:
  1. Country value consistency across the three sources (a mismatch like
     "US" vs "USA" would silently kill every (country, token) lookup).
  2. For a random sample of ground-truth Source 1 entities, every missed
     true pair is put into exactly one failure bucket:
       - country_mismatch      : S1 and candidate have different country values
       - no_shared_name_token  : names share zero significant tokens
       - shared_only_dropped   : they share tokens, but all of them were
                                 dropped as too generic (bucket > cap)
       - shared_kept_truncated : they share a kept token, so the pair WAS
                                 reachable, but got cut by the top-N cap
     plus whether an address token / name character-trigram would have
     caught it.
  3. Real name/address examples for each bucket.
"""
import csv
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from normalize import normalize_business_name, normalize_address, significant_tokens
from blocking import MAX_TOKEN_BUCKET_SIZE, MAX_CANDIDATES_PER_ENTITY

TRAIN = "dataset/train/"
SAMPLE_SIZE = 20000
EXAMPLES_PER_BUCKET = 12
csv.field_size_limit(sys.maxsize)


def trigrams(s):
    s = s.replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main(cand_path):
    rng = random.Random(42)

    # --- ground truth sample ---
    gt = {}
    with open(TRAIN + "train_ground_truth.tsv", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) > 1 and row[1]:
                gt[row[0]] = row[1].split(",")
    sample_ids = set(rng.sample(sorted(gt), min(SAMPLE_SIZE, len(gt))))
    need = set(sample_ids)
    for s1 in sample_ids:
        need.update(gt[s1])
    sizes = Counter(len(v) for v in gt.values())
    print(f"Ground truth: {len(gt):,} S1 entities with >=1 match, "
          f"{sum(len(v) for v in gt.values()):,} true pairs")
    print("  matches per S1 entity (top):", sizes.most_common(8))

    # --- stream sources: country counts, name/address token doc-freq, sampled records ---
    recs = {}
    country_counts = {}
    name_df = Counter()
    addr_df = Counter()
    for src in ("1", "2", "3"):
        cc = Counter()
        with open(f"{TRAIN}train_source{src}.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                c = row.get("country", "")
                cc[c] += 1
                if src != "1":
                    nm = normalize_business_name(row.get("business_name", ""))
                    for t in set(significant_tokens(nm)):
                        name_df[(c, t)] += 1
                    ad = normalize_address(row.get("business_address", ""))
                    for t in set(significant_tokens(ad, min_len=2)):
                        addr_df[(c, t)] += 1
                if row["entity_id"] in need:
                    recs[row["entity_id"]] = row
        country_counts[src] = cc

    print("\n=== Country values per source ===")
    all_c = set().union(*country_counts.values())
    print(f"{'country':>20} {'S1':>10} {'S2':>10} {'S3':>10}")
    for c in sorted(all_c, key=lambda c: -sum(cc[c] for cc in country_counts.values()))[:25]:
        print(f"{c!r:>20} " + " ".join(f"{country_counts[s][c]:>10,}" for s in "123"))

    # --- candidates for sample ---
    cands = {}
    with open(cand_path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in sample_ids:
                cands[row[0]] = row[1].split(",") if len(row) > 1 and row[1] else []

    # --- categorize every true pair in the sample ---
    total = found = 0
    bucket = Counter()
    bucket_ex = defaultdict(list)
    zero_cand_miss = some_cand_miss = 0
    addr_rescue = Counter()
    tri_hist = Counter()
    miss_by_src = Counter()
    miss_by_country = Counter()
    s1_zero = sum(1 for s in sample_ids if not cands.get(s))
    for s1 in sample_ids:
        a = recs.get(s1)
        cl = cands.get(s1, [])
        cs = set(cl)
        if a is None:
            continue
        a_c = a.get("country", "")
        a_tok = set(significant_tokens(normalize_business_name(a["business_name"])))
        a_adr = set(significant_tokens(normalize_address(a.get("business_address", "")), min_len=2))
        a_tri = trigrams(normalize_business_name(a["business_name"]))
        for m in gt[s1]:
            total += 1
            if m in cs:
                found += 1
                continue
            b = recs.get(m)
            if b is None:
                bucket["missing_record"] += 1
                continue
            miss_by_src[m[:2]] += 1
            miss_by_country[a_c] += 1
            if cl:
                some_cand_miss += 1
            else:
                zero_cand_miss += 1
            b_c = b.get("country", "")
            b_tok = set(significant_tokens(normalize_business_name(b["business_name"])))
            shared = a_tok & b_tok
            kept = {t for t in shared if name_df[(a_c, t)] <= MAX_TOKEN_BUCKET_SIZE}
            if a_c != b_c:
                cat = "country_mismatch"
            elif not shared:
                cat = "no_shared_name_token"
            elif not kept:
                cat = "shared_only_dropped"
            else:
                cat = "shared_kept_truncated"
            bucket[cat] += 1
            b_adr = set(significant_tokens(normalize_address(b.get("business_address", "")), min_len=2))
            sa = a_adr & b_adr
            if not a_adr or not b_adr:
                addr_rescue[(cat, "addr_empty")] += 1
            elif any(addr_df[(a_c, t)] <= MAX_TOKEN_BUCKET_SIZE for t in sa):
                addr_rescue[(cat, "addr_rare_shared")] += 1
            elif sa:
                addr_rescue[(cat, "addr_only_generic_shared")] += 1
            else:
                addr_rescue[(cat, "addr_nothing_shared")] += 1
            j = jaccard(a_tri, trigrams(normalize_business_name(b["business_name"])))
            tri_hist[(cat, min(int(j * 10), 9))] += 1
            if len(bucket_ex[cat]) < EXAMPLES_PER_BUCKET:
                bucket_ex[cat].append((a, b, shared, kept, j,
                                       [(t, name_df[(a_c, t)]) for t in shared]))

    miss = total - found
    print(f"\n=== Sample of {len(sample_ids):,} S1 entities ===")
    print(f"True pairs: {total:,}  found: {found:,} ({found / total:.1%})  missed: {miss:,}")
    print(f"S1 entities with ZERO candidates: {s1_zero:,} ({s1_zero / len(sample_ids):.1%})")
    print(f"Missed pairs where S1 had zero candidates: {zero_cand_miss:,}; had some (wrong) candidates: {some_cand_miss:,}")
    print(f"Missed by source: {dict(miss_by_src)}")
    print(f"Missed by S1 country (top): {miss_by_country.most_common(8)}")
    print("\nFailure buckets (share of missed pairs):")
    for k, v in bucket.most_common():
        print(f"  {k:25s} {v:>8,}  {v / miss:.1%}")
    print("\nAddress signal for missed pairs:")
    for k, v in sorted(addr_rescue.items()):
        print(f"  {k[0]:25s} {k[1]:26s} {v:>8,}")
    print("\nName char-trigram Jaccard for missed pairs (decile: count):")
    for cat in bucket:
        row = [tri_hist[(cat, d)] for d in range(10)]
        print(f"  {cat:25s} {row}")

    for cat, exs in bucket_ex.items():
        print(f"\n--- Examples: {cat} ---")
        for a, b, shared, kept, j, dfs in exs:
            print(f"  S1 [{a['country']}] {a['business_name']!r} | {a.get('business_address', '')!r}")
            print(f"  {b['entity_id'][:2]} [{b['country']}] {b['business_name']!r} | {b.get('business_address', '')!r}")
            print(f"     shared={dfs} kept={sorted(kept)} trigramJ={j:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/candidate_pairs_train.tsv")
