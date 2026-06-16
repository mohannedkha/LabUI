# jfr — Journal-Fit Recommender and Submission Tracker

A local-first tool for routing manuscripts to the right journals and tracking every submission lifecycle event — from first draft to accepted (or rejected and re-routed).

**Problem it solves:** Scope mismatches detected *before* submission rather than after a 2–4 month desk rejection. The recommender scores your manuscript abstract against the journal's *actual recent acceptance pattern* (not its scope statement), under your policy constraints.

---

## Quick start

```bash
# Activate the environment
source .venv/bin/activate

# One-time: seed journal metadata into the database
jfr corpus init

# Fetch recent abstracts from OpenAlex (one journal or all)
jfr corpus refresh                          # all 15 journals
jfr corpus refresh langmuir --limit 500     # single journal

# Check what was fetched
jfr corpus stats
jfr corpus stats langmuir

# Build vector embeddings (downloads specter2 ~500 MB on first run)
jfr corpus embed

# Add your manuscript abstract
jfr manuscript add path/to/abstract.md \
  --title "Your paper title" \
  --claim "One-sentence principal claim." \
  --techniques "NTA,zeta potential,XDLVO,molecular dynamics"

# Get ranked journal recommendations
jfr recommend --manuscript ms_<id> --top 10
jfr recommend --manuscript ms_<id> --top 10 --explain   # show score breakdown

# Create and advance a submission record
jfr submission create --manuscript ms_<id> --journal langmuir
jfr submission transition <sub_id> internal_review --note "Sent to co-author"
jfr submission transition <sub_id> awaiting_supervisor_approval
jfr submission transition <sub_id> submitted
jfr submission transition <sub_id> under_review

# See everything in flight
jfr view active
```

---

## Installation

Requires nothing beyond this folder — the Python 3.12 environment is self-contained.

```bash
# If uv is not on PATH yet (only needed once per shell session):
export PATH="$HOME/.local/bin:$PATH"

# Activate
source .venv/bin/activate

# Verify
jfr --version      # prints 0.1.0
```

To make activation permanent, add this to `~/.bashrc` or `~/.zshrc`:

```bash
alias jfr-activate='source /path/to/LabUI/submission_strategy/.venv/bin/activate'
```

---

## Command reference

### `jfr corpus`

| Command | What it does |
|---|---|
| `jfr corpus init` | Load `data/journals.yaml` into the SQLite database |
| `jfr corpus refresh [JOURNAL_ID]` | Fetch abstracts from OpenAlex (or `--source crossref`) |
| `jfr corpus stats [JOURNAL_ID]` | Print article counts, date range, top topics |
| `jfr corpus embed [JOURNAL_ID]` | Compute specter2 embeddings and store in Qdrant |
| `jfr corpus status` | Show embedding coverage (vectorised / total) per journal |

**Options for `refresh`:**

| Flag | Default | Description |
|---|---|---|
| `--source` | `openalex` | `openalex` or `crossref` |
| `--months` | `36` | Corpus window |
| `--limit` | `500` | Max articles per journal |
| `--embed` | off | Also run embedder immediately after fetch |

---

### `jfr manuscript`

| Command | What it does |
|---|---|
| `jfr manuscript add <path>` | Add manuscript from a plain-text or markdown abstract file |
| `jfr manuscript list` | List all manuscripts |
| `jfr manuscript show <ms_id>` | Full manuscript record as JSON |

**Options for `add`:**

| Flag | Description |
|---|---|
| `--title TEXT` | Override title (default: derived from filename) |
| `--claim TEXT` | Principal claim — one declarative sentence |
| `--techniques TEXT` | Comma-separated technique tags: `"NTA,zeta potential,XDLVO"` |
| `--bibtex TEXT` | BibTeX key for cross-reference with Zotero |
| `--ms-id TEXT` | Set an explicit ID (default: auto-generated) |

---

### `jfr recommend`

```
jfr recommend --manuscript <ms_id> [--top N] [--explain]
```

Returns a ranked table of journals. Each entry shows:
- Scope-fit score (0–1) decomposed into abstract / lexical / claim sub-scores
- Policy status (pass / fail with reason)
- Your prior history at that venue (acceptances, rejections)
- Rationale sentence

`--explain` prints the score decomposition row-by-row beneath the table.

> **Note:** Run `jfr corpus embed` first; journals with no vector index score 0 on the dense component but still rank on lexical overlap.  
> The default embedding model is `allenai/specter2_base` (~450 MB, downloads automatically on first run). Set `HF_TOKEN` in your environment for faster HuggingFace downloads.

---

### `jfr submission`

| Command | What it does |
|---|---|
| `jfr submission create --manuscript <id> --journal <id>` | Open a new record in `drafting` state |
| `jfr submission transition <sub_id> <to_state>` | Advance the state machine |
| `jfr submission show <sub_id>` | Full record with transition history as JSON |
| `jfr submission comment <sub_id> --reviewer N --text "..."` | Log a reviewer comment |

**Valid state machine transitions:**

```
drafting
  └─► internal_review
        └─► awaiting_supervisor_approval
              └─► submitted
                    └─► under_review
                          ├─► revision_requested_minor ─┐
                          ├─► revision_requested_major ─┤
                          │                              └─► revising ─► resubmitted ─► (loop)
                          ├─► accepted           (terminal)
                          └─► rejected_post_review (terminal)
                    └─► rejected_desk            (terminal)
  (any non-terminal) ─► withdrawn               (terminal)
```

Illegal transitions are rejected at write time with a clear error message.

---

### `jfr view`

| Command | What it does |
|---|---|
| `jfr view active` | All non-terminal submissions, sorted by time in current state |
| `jfr view manuscript <ms_id>` | All submission attempts for one paper |
| `jfr view journal <journal_id>` | Full history for one journal across all manuscripts |

---

## Web interface

```bash
jfr web serve                        # starts on http://localhost:8765
jfr web serve --port 9000 --reload   # custom port, auto-reload for dev
```

The web UI exposes a REST API under `/api/` (browse at `/api/docs`). The dashboard HTML at `/` is a placeholder for Phase 3 — the full interface is under development.

Key API endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/journals` | List all tracked journals |
| `GET` | `/api/manuscripts` | List all manuscripts |
| `POST` | `/api/manuscripts` | Add a manuscript |
| `GET` | `/api/recommend/{ms_id}?top=10` | Get ranked recommendations |
| `GET` | `/api/submissions/active` | Active submissions |
| `POST` | `/api/submissions/{sub_id}/transition` | Advance state machine |

---

## Configuring your policy

Edit `data/policy.toml` before running recommendations:

```toml
[open_access]
fully_oa_allowed = false   # set true to include fully-OA venues
hybrid_oa_allowed = true

[publishers]
blocklist = []
preferlist = ["acs", "elsevier", "wiley", "rsc"]

[quality]
impact_factor_floor = 4.0  # journals below this IF are filtered out

[approvals]
require_supervisor_approval = true
supervisor_contact = "supervisor@university.edu"

[exclusions]
journals = [
    { id = "chemosphere", reason = "retraction_concerns" },
]
```

Policy constraints are **hard filters**: excluded journals appear in the output with `policy: fail` and a reason, but are never ranked.

---

## Adding journals

Edit `data/journals.yaml` and add a new entry following the existing format, then re-run:

```bash
jfr corpus init       # re-seeds all journals from YAML
jfr corpus refresh <new_journal_id>
jfr corpus embed <new_journal_id>
```

Minimum required fields: `id`, `name`, `publisher`, `publisher_family`, `issn_electronic` (or `issn_print`), `is_fully_oa`, `is_hybrid_oa`.

---

## Where data lives

All persistent data is in `~/.local/share/jfr/`:

| Path | Contents |
|---|---|
| `db.sqlite` | All manuscripts, journals, corpus articles, submissions |
| `vectors/` | Qdrant embedded vector store (one collection per journal) |
| `attachments/` | Decision letters, cover letters, highlights |
| `models/` | (sentence-transformers caches here via HuggingFace) |

No manuscript text ever leaves your machine. Corpus refresh sends only ISSNs and DOIs to public APIs (CrossRef, OpenAlex).

---

## Nightly corpus refresh (optional)

To keep the corpus fresh automatically, create a systemd user timer:

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/jfr-refresh.service << 'EOF'
[Unit]
Description=jfr nightly corpus refresh

[Service]
Type=oneshot
ExecStart=/path/to/LabUI/submission_strategy/.venv/bin/jfr corpus refresh --embed
Environment=JFR_DATA_DIR=%h/.local/share/jfr
EOF

cat > ~/.config/systemd/user/jfr-refresh.timer << 'EOF'
[Unit]
Description=Run jfr corpus refresh nightly

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now jfr-refresh.timer
systemctl --user list-timers jfr-refresh.timer
```

---

## GPU acceleration (AMD Radeon 8060S / ROCm)

The embedding step runs on CPU by default. To use the GPU:

```bash
# Install ROCm-enabled PyTorch (replaces the CPU build)
# Check https://pytorch.org/get-started/locally/ for the current ROCm URL
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall

# Verify
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

sentence-transformers will automatically pick up the GPU. Embedding throughput on the Radeon 8060S is ~800 docs/min vs ~80 docs/min on CPU.

**Embedding model note:** The default model is `allenai/specter2_base`. To upgrade to the full `allenai/specter2` (LoRA adapters, marginally higher accuracy), pin `peft` to a version matching the model's config and delete `~/.local/share/jfr/vectors/` to force re-indexing.

---

## Data backup

```bash
# Dump the SQLite database to portable SQL, then snapshot with restic
sqlite3 ~/.local/share/jfr/db.sqlite .dump > ~/backups/jfr_$(date +%Y%m%d).sql

# Or add to your existing restic repo (excludes model weights and vector store,
# which are regenerable from the SQL dump + CrossRef/OpenAlex)
restic backup ~/.local/share/jfr \
  --exclude ~/.local/share/jfr/vectors \
  --exclude ~/.local/share/jfr/models
```

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Corpus builder, SQLite schema, 15 journals, CLI skeleton | **Done** |
| 2 | Embedding pipeline, hybrid scoring, `jfr recommend`, offline validation | **Done** |
| 3 | Web UI on `localhost:8765`, stall detection, RSS refresh | **Done** |
| 4 | Obsidian vault integration, email decision-letter parsing | Planned |

---

## Journal IDs (pre-seeded)

| ID | Journal |
|---|---|
| `langmuir` | Langmuir (ACS) |
| `jcis` | Journal of Colloid and Interface Science (Elsevier) |
| `colsurfa` | Colloids and Surfaces A (Elsevier) |
| `acsnano` | ACS Nano (ACS) |
| `cej` | Chemical Engineering Journal (Elsevier) |
| `softmatter` | Soft Matter (RSC) |
| `energyfuels` | Energy & Fuels (ACS) |
| `jpcb` | Journal of Physical Chemistry B (ACS) |
| `nanoscale` | Nanoscale (RSC) |
| `colsurfb` | Colloids and Surfaces B (Elsevier) |
| `ijhmt` | International Journal of Heat and Mass Transfer (Elsevier) |
| `pccp` | Physical Chemistry Chemical Physics (RSC) |
| `jmca` | Journal of Materials Chemistry A (RSC) |
| `ces` | Chemical Engineering Science (Elsevier) |
| `jiec` | Journal of Industrial and Engineering Chemistry (Elsevier) |
