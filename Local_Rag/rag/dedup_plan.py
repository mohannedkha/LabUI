#!/usr/bin/env python3
"""Content-based duplicate detection for the RAG library.

Clusters papers by EXACT normalized-text hash + NEAR-duplicate Jaccard
(8-word shingles). Picks one canonical per cluster (most chunks => most
complete parse; tie-break: source PDF exists, then shorter paper_id).
Writes data/dedup_manifest.json. Removes NOTHING.
"""
import sqlite3, collections, hashlib, re, os, json
from pathlib import Path

ROOT = Path(__file__).parent
DB = ROOT / "data" / "rag.db"
PARSED = ROOT / "data" / "parsed"
NEAR_THRESHOLD = 0.80   # >=0.80 auto-clustered; 0.70-0.80 flagged for review
REVIEW_LOW = 0.70

c = sqlite3.connect(str(DB))
meta = {pid: (title, src) for pid, title, src in
        c.execute("select paper_id,title,source_pdf from papers")}
nchunks = dict(c.execute("select paper_id,count(*) from chunks group by paper_id"))
rows = c.execute("select paper_id,text from chunks order by paper_id,position").fetchall()

txt = collections.defaultdict(list)
for pid, t in rows:
    txt[pid].append(t or "")

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

full = {pid: norm(" ".join(v)) for pid, v in txt.items()}

def shingles(s, k=8):
    w = s.split()
    return set(hashlib.md5(" ".join(w[i:i+k]).encode()).hexdigest()
               for i in range(0, max(1, len(w)-k+1)))

sh = {pid: shingles(s) for pid, s in full.items()}
wc = {pid: len(s.split()) for pid, s in full.items()}
pids = list(full)

# union-find
parent = {p: p for p in pids}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b);
    if ra != rb: parent[ra] = rb

# exact hash edges
h = collections.defaultdict(list)
for pid, s in full.items():
    h[hashlib.md5(s.encode()).hexdigest()].append(pid)
for grp in h.values():
    for x in grp[1:]:
        union(grp[0], x)

# near-dup edges
review_pairs = []
for i in range(len(pids)):
    a = pids[i]
    for j in range(i+1, len(pids)):
        b = pids[j]
        if not sh[a] or not sh[b]: continue
        if min(wc[a], wc[b]) / max(wc[a], wc[b]) < 0.6: continue
        inter = len(sh[a] & sh[b])
        if not inter: continue
        jac = inter / len(sh[a] | sh[b])
        if jac >= NEAR_THRESHOLD:
            union(a, b)
        elif jac >= REVIEW_LOW:
            review_pairs.append((round(jac, 3), a, b))

clusters = collections.defaultdict(list)
for p in pids:
    clusters[find(p)].append(p)
dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

def pdf_exists(pid):
    src = meta.get(pid, ("", ""))[1] or ""
    return os.path.exists(src)

def good_title(pid):
    t = (meta.get(pid, ("", ""))[0] or "").lower()
    bad = ("view article online", "journal", "licensed under", "downloaded",
           "issn", "doi.org", "see paper", "repository", "proceedings")
    return 0 if (not t or any(b in t for b in bad)) else 1

def canonical(grp):
    # most chunks, then pdf exists, then real title, then shortest id
    return sorted(grp, key=lambda p: (-nchunks.get(p, 0), -int(pdf_exists(p)),
                                      -good_title(p), len(p)))[0]

manifest = {"keep": [], "remove": [], "clusters": [], "review_pairs": []}
for grp in dup_clusters.values():
    keep = canonical(grp)
    rem = [p for p in grp if p != keep]
    manifest["keep"].append(keep)
    manifest["remove"].extend(rem)
    manifest["clusters"].append({
        "keep": {"paper_id": keep, "chunks": nchunks.get(keep, 0),
                 "pdf": meta.get(keep, ("", ""))[1]},
        "remove": [{"paper_id": p, "chunks": nchunks.get(p, 0),
                    "pdf": meta.get(p, ("", ""))[1]} for p in rem],
    })
manifest["review_pairs"] = [
    {"jaccard": j, "a": a, "b": b,
     "a_pdf": meta.get(a, ("", ""))[1], "b_pdf": meta.get(b, ("", ""))[1]}
    for j, a, b in sorted(review_pairs, reverse=True)]

(ROOT / "data" / "dedup_manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"Total papers: {len(pids)}")
print(f"Duplicate clusters: {len(dup_clusters)}")
print(f"Papers to KEEP (canonicals of dup clusters): {len(manifest['keep'])}")
print(f"Papers to REMOVE (redundant copies): {len(manifest['remove'])}")
print(f"After dedup: {len(pids) - len(manifest['remove'])} unique papers")
print(f"Borderline pairs for manual review (0.70-0.80): {len(manifest['review_pairs'])}")
print("\nManifest written: data/dedup_manifest.json")
print("\n--- CLUSTERS (keep <= remove) ---")
for cl in manifest["clusters"]:
    k = cl["keep"]
    print(f"KEEP  {k['paper_id']}  ({k['chunks']} chunks)")
    for r in cl["remove"]:
        print(f"  rm  {r['paper_id']}  ({r['chunks']} chunks)")
if manifest["review_pairs"]:
    print("\n--- BORDERLINE (review, NOT auto-removed) ---")
    for rp in manifest["review_pairs"]:
        print(f"  {rp['jaccard']}  {rp['a']}  ||  {rp['b']}")
