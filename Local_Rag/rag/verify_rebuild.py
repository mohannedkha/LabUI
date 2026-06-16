#!/usr/bin/env python3
"""Post-rebuild integrity check. Prints a report; exits non-zero on hard errors."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_PATH, DATA_DIR

GRAPH_DB = DATA_DIR / "graph.db"
errors = []


def open_vec(path):
    import sqlite_vec
    c = sqlite3.connect(str(path))
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    c.row_factory = sqlite3.Row
    return c


c = open_vec(DB_PATH)
papers = c.execute("select count(*) from papers").fetchone()[0]
chunks = c.execute("select count(*) from chunks").fetchone()[0]
vecs = c.execute("select count(*) from chunks_vec").fetchone()[0]
fts = c.execute("select count(*) from chunks_fts").fetchone()[0]
print(f"papers={papers}  chunks={chunks}  vectors={vecs}  fts={fts}")

# orphan checks
o_chunks = c.execute("select count(*) from chunks where paper_id not in (select paper_id from papers)").fetchone()[0]
ch_ids = set(r[0] for r in c.execute("select chunk_id from chunks"))
vec_ids = set(r[0] for r in c.execute("select chunk_id from chunks_vec"))
print(f"chunks w/o paper={o_chunks}  vectors w/o chunk={len(vec_ids-ch_ids)}  chunks w/o vector={len(ch_ids-vec_ids)}")
if o_chunks or (vec_ids - ch_ids) or (ch_ids - vec_ids):
    errors.append("rag.db store misalignment")
if not (chunks == vecs == fts):
    errors.append(f"count mismatch chunks/vecs/fts = {chunks}/{vecs}/{fts}")

# vector dim
dim_row = c.execute("select value from chunks_vec_info where key='dimensions'").fetchone() if \
    c.execute("select count(*) from sqlite_master where name='chunks_vec_info'").fetchone()[0] else None
if dim_row:
    print(f"vector dim (from chunks_vec_info): {dim_row[0]}")

# title quality sample
junky = c.execute(
    "select count(*) from papers where lower(title) like '%view article online%' "
    "or lower(title) like '%is an international journal%' or lower(title) like 'doi%' "
    "or title is null or length(title)<6"
).fetchone()[0]
print(f"papers with junk/empty title: {junky}")

# page-number sanity: how many chunks have page_start>1 (proves real pages, not all 1)
pages_gt1 = c.execute("select count(*) from chunks where page_start>1").fetchone()[0]
print(f"chunks with page_start>1 (real pages): {pages_gt1}")

print("\n--- sample papers (title | pages of first chunk) ---")
for r in c.execute(
    "select p.paper_id,p.title, (select min(page_start) from chunks where paper_id=p.paper_id) ps,"
    " (select max(page_end) from chunks where paper_id=p.paper_id) pe from papers p limit 8"):
    print(f"  {r['title'][:70]!r}  pp.{r['ps']}-{r['pe']}")

# graph alignment
if GRAPH_DB.exists():
    g = sqlite3.connect(str(GRAPH_DB))
    ge = g.execute("select count(*) from entities").fetchone()[0]
    gr = g.execute("select count(*) from relations").fetchone()[0]
    try:
        gec_ids = set(r[0] for r in g.execute("select distinct chunk_id from entity_chunks"))
        dangling = len(gec_ids - ch_ids)
    except Exception:
        dangling = -1
    print(f"\ngraph: entities={ge} relations={gr} entity_chunk chunk_ids dangling (not in rag.db)={dangling}")
    if dangling > 0:
        errors.append(f"graph has {dangling} dangling chunk_ids")
    g.close()

c.close()
print()
if errors:
    print("VERIFY: FAILED ->", "; ".join(errors))
    sys.exit(1)
print("VERIFY: OK")
