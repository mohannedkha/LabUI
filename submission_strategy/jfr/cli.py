"""jfr — Journal-Fit Recommender CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from jfr.config import get_settings, load_policy
from jfr.db import init_db, get_conn

app = typer.Typer(
    name="jfr",
    help="Journal-Fit Recommender and Submission Tracker",
    no_args_is_help=True,
)
corpus_app = typer.Typer(help="Corpus management commands", no_args_is_help=True)
manuscript_app = typer.Typer(help="Manuscript management", no_args_is_help=True)
submission_app = typer.Typer(help="Submission tracking", no_args_is_help=True)
view_app = typer.Typer(help="Aggregated views", no_args_is_help=True)

app.add_typer(corpus_app, name="corpus")
app.add_typer(manuscript_app, name="manuscript")
app.add_typer(submission_app, name="submission")
app.add_typer(view_app, name="view")

# Add RAG integration sub-app
rag_app = typer.Typer(help="RAG integration commands (search, link, lab)", no_args_is_help=True)
app.add_typer(rag_app, name="rag")

# Rag sub-typer for search operations
rag_search_app = typer.Typer(help="RAG search commands", no_args_is_help=True)
rag_app.add_typer(rag_search_app, name="search")

# Rag sub-typer for linking operations
rag_link_app = typer.Typer(help="RAG linking commands", no_args_is_help=True)
rag_app.add_typer(rag_link_app, name="link")

# Rag sub-typer for lab operations
rag_lab_app = typer.Typer(help="Research lab commands", no_args_is_help=True)
rag_app.add_typer(rag_lab_app, name="lab")

console = Console()


def _get_conn():
    s = get_settings()
    s.ensure_dirs()
    return init_db(s.db_path)


def _is_tty() -> bool:
    return sys.stdout.isatty()


# ── corpus ────────────────────────────────────────────────────────────────────

@corpus_app.command("init")
def corpus_init():
    """Load journals from data/journals.yaml into the database."""
    from jfr.tracker import load_journals_from_yaml
    conn = _get_conn()
    s = get_settings()
    count = load_journals_from_yaml(conn, s.journals_yaml)
    rprint(f"[green]✓[/green] Loaded {count} journals")


@corpus_app.command("refresh")
def corpus_refresh(
    journal_id: Optional[str] = typer.Argument(None, help="Journal ID to refresh (all if omitted)"),
    source: str = typer.Option("openalex", help="Data source: crossref or openalex"),
    months: int = typer.Option(36, help="Corpus window in months"),
    limit: int = typer.Option(500, help="Max articles per journal"),
    embed: bool = typer.Option(False, help="Also compute embeddings after ingestion"),
):
    """Fetch recent abstracts from CrossRef or OpenAlex and ingest into corpus."""
    from jfr.corpus import fetch_crossref_articles, fetch_openalex_articles, ingest_articles
    conn = _get_conn()
    s = get_settings()

    if journal_id:
        journals = [dict(conn.execute("SELECT * FROM journal WHERE id=?", (journal_id,)).fetchone())]
    else:
        journals = [dict(r) for r in conn.execute("SELECT * FROM journal").fetchall()]

    if not journals:
        rprint("[red]No journals found. Run `jfr corpus init` first.[/red]")
        raise typer.Exit(1)

    for j in journals:
        jid = j["id"]
        issn = j.get("issn_electronic") or j.get("issn_print")
        if not issn:
            rprint(f"[yellow]⚠[/yellow]  {jid}: no ISSN, skipping")
            continue
        rprint(f"[cyan]→[/cyan] Fetching {jid} ({issn}) from {source}…")
        try:
            if source == "crossref":
                articles = fetch_crossref_articles(issn, months=months, limit=limit)
            elif source == "rss":
                meta = json.loads(j.get("metadata_json") or "{}")
                feed_url = meta.get("rss_feed_url")
                if not feed_url:
                    rprint(f"[yellow]⚠[/yellow]  {jid}: no rss_feed_url in metadata, skipping")
                    continue
                from jfr.corpus import fetch_rss_articles
                articles = fetch_rss_articles(feed_url)
            else:
                articles = fetch_openalex_articles(issn, months=months, limit=limit)
        except Exception as e:
            rprint(f"[red]  Error:[/red] {e}")
            continue
        n = ingest_articles(conn, jid, articles)
        rprint(f"[green]  ✓[/green] {jid}: {n} new articles ingested ({len(articles)} fetched)")

        if embed:
            _do_embed(conn, jid, s)


@corpus_app.command("stats")
def corpus_stats_cmd(
    journal_id: Optional[str] = typer.Argument(None, help="Journal ID (all if omitted)"),
):
    """Print corpus statistics for one or all journals."""
    from jfr.corpus import corpus_stats
    conn = _get_conn()

    if journal_id:
        journal_ids = [journal_id]
    else:
        journal_ids = [r["id"] for r in conn.execute("SELECT id FROM journal ORDER BY id").fetchall()]

    if not journal_ids:
        rprint("[yellow]No journals in database. Run `jfr corpus init` first.[/yellow]")
        return

    if _is_tty():
        table = Table(title="Corpus Statistics", show_lines=True)
        table.add_column("Journal ID", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("w/ Abstract", justify="right")
        table.add_column("Oldest")
        table.add_column("Newest")
        table.add_column("Top Topics")
        for jid in journal_ids:
            stats = corpus_stats(conn, jid)
            topics_str = ", ".join(t for t, _ in stats["top_topics"][:3]) or "—"
            table.add_row(
                jid,
                str(stats["total"]),
                str(stats["with_abstract"]),
                stats["oldest"] or "—",
                stats["newest"] or "—",
                topics_str,
            )
        console.print(table)
    else:
        out = {}
        for jid in journal_ids:
            out[jid] = corpus_stats(conn, jid)
        print(json.dumps(out, indent=2))


@corpus_app.command("status")
def corpus_status():
    """Show embedding coverage per journal (articles with vectors vs total)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT journal_id,
                  COUNT(*) as total,
                  SUM(CASE WHEN vector_id IS NOT NULL THEN 1 ELSE 0 END) as vectorised
           FROM corpus_article GROUP BY journal_id ORDER BY journal_id"""
    ).fetchall()
    if _is_tty():
        table = Table(title="Embedding Status")
        table.add_column("Journal", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Vectorised", justify="right")
        table.add_column("Coverage", justify="right")
        for r in rows:
            pct = r["vectorised"] / r["total"] * 100 if r["total"] else 0
            color = "green" if pct == 100 else "yellow" if pct > 0 else "red"
            table.add_row(r["journal_id"], str(r["total"]), str(r["vectorised"]),
                          f"[{color}]{pct:.0f}%[/{color}]")
        console.print(table)
    else:
        print(json.dumps([dict(r) for r in rows], indent=2))


@corpus_app.command("embed")
def corpus_embed(
    journal_id: Optional[str] = typer.Argument(None, help="Journal ID to embed (all if omitted)"),
):
    """Compute and store embeddings for corpus articles in Qdrant."""
    conn = _get_conn()
    s = get_settings()

    if journal_id:
        journal_ids = [journal_id]
    else:
        journal_ids = [r["id"] for r in conn.execute("SELECT id FROM journal ORDER BY id").fetchall()]

    for jid in journal_ids:
        _do_embed(conn, jid, s)


@corpus_app.command("update-if")
def corpus_update_if(
    journal_id: Optional[str] = typer.Argument(None, help="Journal ID to update (all if omitted)"),
):
    """Update impact factors from OpenAlex API (uses 2yr_mean_citedness as proxy)."""
    from jfr.corpus import fetch_crossref_articles  # just for import check
    conn = _get_conn()
    s = get_settings()

    if journal_id:
        journals = [dict(conn.execute("SELECT id, issn_electronic, issn_print FROM journal WHERE id=?", (journal_id,)).fetchone())]
    else:
        journals = [dict(r) for r in conn.execute("SELECT id, issn_electronic, issn_print FROM journal").fetchall()]

    if not journals:
        rprint("[red]No journals found. Run `jfr corpus init` first.[/red]")
        raise typer.Exit(1)

    import httpx
    import time as _time

    OPENALEX_API = "https://api.openalex.org/sources"
    UA = "jfr/0.1"

    updated = 0
    with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
        for j in journals:
            jid = j["id"]
            issn = j.get("issn_electronic") or j.get("issn_print")
            if not issn:
                rprint(f"[yellow]⚠[/yellow]  {jid}: no ISSN, skipping")
                continue
            rprint(f"[cyan]→[/cyan] Fetching IF for {jid} ({issn})…")
            try:
                resp = client.get(OPENALEX_API, params={"filter": f"issn:{issn}", "per-page": 1})
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if results:
                    source = results[0]
                    summary = source.get("summary_stats", {})
                    new_if = summary.get("2yr_mean_citedness", 0)
                    if new_if:
                        conn.execute("UPDATE journal SET impact_factor=? WHERE id=?", (round(new_if, 1), jid))
                        updated += 1
                        rprint(f"[green]  ✓[/green] {jid}: IF updated to {round(new_if, 1)}")
                    else:
                        rprint(f"[yellow]  ⚠[/yellow]  {jid}: no IF available")
                else:
                    rprint(f"[yellow]  ⚠[/yellow]  {jid}: no source found")
            except Exception as e:
                rprint(f"[red]  Error:[/red] {e}")
            _time.sleep(0.5)  # Rate limit protection

    conn.commit()
    rprint(f"\n[green]✓[/green] Updated {updated} impact factors")


def _do_embed(conn, jid: str, s):
    from jfr.corpus.embedder import embed_corpus
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    qc = QdrantClient(path=str(s.vectors_dir))
    collection = f"journal_{jid}"
    try:
        qc.get_collection(collection)
    except Exception:
        qc.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    rprint(f"[cyan]→[/cyan] Embedding {jid}…")
    n = embed_corpus(
        conn, jid, qc, collection, s.abstract_model,
        progress_cb=lambda done, total: rprint(f"   {done}/{total}", end="\r"),
    )
    rprint(f"[green]  ✓[/green] {jid}: {n} articles embedded")


# ── manuscript ────────────────────────────────────────────────────────────────

@manuscript_app.command("add")
def manuscript_add(
    path: Path = typer.Argument(..., help="Path to abstract markdown file"),
    title: Optional[str] = typer.Option(None, help="Manuscript title"),
    claim: Optional[str] = typer.Option(None, help="Principal claim (one sentence)"),
    techniques: Optional[str] = typer.Option(None, help="Comma-separated technique tags"),
    bibtex: Optional[str] = typer.Option(None, help="BibTeX key"),
    ms_id: Optional[str] = typer.Option(None, help="Explicit manuscript ID"),
):
    """Add a manuscript from an abstract markdown file."""
    from jfr.tracker import create_manuscript
    conn = _get_conn()

    abstract = path.read_text().strip()
    title_ = title or path.stem.replace("_", " ").replace("-", " ").title()
    claim_ = claim or abstract.split(".")[0] + "."
    techs = [t.strip() for t in techniques.split(",")] if techniques else []

    ms_id_ = create_manuscript(
        conn, title_, abstract, claim_, techniques=techs, bibtex_key=bibtex, ms_id=ms_id
    )
    rprint(f"[green]✓[/green] Added [bold]{ms_id_}[/bold]")


@manuscript_app.command("list")
def manuscript_list():
    """List all manuscripts."""
    from jfr.tracker import list_manuscripts
    conn = _get_conn()
    mss = list_manuscripts(conn)
    if _is_tty():
        table = Table(title="Manuscripts")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Created")
        for m in mss:
            table.add_row(m["id"], m["title"][:60], m["created_at"][:10])
        console.print(table)
    else:
        print(json.dumps(mss, indent=2))


@manuscript_app.command("show")
def manuscript_show(ms_id: str):
    """Show manuscript details."""
    from jfr.tracker import get_manuscript
    conn = _get_conn()
    m = get_manuscript(conn, ms_id)
    if not m:
        rprint(f"[red]Manuscript {ms_id!r} not found[/red]")
        raise typer.Exit(1)
    print(json.dumps(m, indent=2))


# ── recommend ─────────────────────────────────────────────────────────────────

@app.command("recommend")
def recommend_cmd(
    manuscript_id: str = typer.Option(..., "--manuscript", "-m"),
    top: int = typer.Option(10, "--top", "-n"),
    explain: bool = typer.Option(False, "--explain", help="Show score decomposition"),
    deliver: str = typer.Option("", "--deliver", help="Delivery target: whatsapp, local, or origin"),
):
    """Rank journals for a manuscript."""
    from jfr.tracker import get_manuscript
    from jfr.matching import ManuscriptInput, recommend
    from qdrant_client import QdrantClient

    conn = _get_conn()
    s = get_settings()
    policy = load_policy(s.policy_toml)

    ms = get_manuscript(conn, manuscript_id)
    if not ms:
        rprint(f"[red]Manuscript {manuscript_id!r} not found[/red]")
        raise typer.Exit(1)

    import json as _json
    inp = ManuscriptInput(
        title=ms["title"],
        abstract=ms["abstract"],
        principal_claim=ms["principal_claim"],
        techniques=_json.loads(ms["techniques_json"]),
        figures=_json.loads(ms["figures_json"]),
    )

    qc = QdrantClient(path=str(s.vectors_dir))
    rprint("[cyan]→[/cyan] Running recommendation engine…")
    results = recommend(
        inp, conn, qc, policy,
        s.abstract_model, s.claim_model,
        top_n=top, manuscript_id=manuscript_id,
    )

    if deliver == "whatsapp":
        # Format as concise WhatsApp message
        msg_lines = [
            f"📊 JFR Recommendation for {ms['title'][:50]}",
            f"Manuscript: {manuscript_id}",
            "",
        ]
        for r in results:
            if r.policy.passed:
                policy_str = "✓" if r.policy.passed else "✗"
                hist = r.user_history
                hist_str = f"acc:{hist.get('prior_acceptances',0)} rej:{hist.get('prior_rejections',0)}"
                msg_lines.append(
                    f"  {r.rank}. {r.journal_name[:40]} — score {r.score:.3f} {policy_str} ({hist_str})"
                )
                if r.rationale:
                    msg_lines.append(f"     {r.rationale[:80]}")
        msg_lines.append("")
        msg_lines.append("Run `jfr recommend -m {manuscript_id}` for full details.")
        whatsapp_msg = "\n".join(msg_lines)

        # Optional push delivery via a user-provided `hermes_tools` module.
        try:
            from hermes_tools import send_message
            send_message(action="send", target="whatsapp", message=whatsapp_msg)
            rprint(f"[green]✓[/green] Recommendations delivered to WhatsApp")
        except ImportError:
            rprint(whatsapp_msg)
            rprint("[dim](push delivery not configured — provide a `hermes_tools` module to enable)[/dim]")
    elif _is_tty():
        table = Table(title=f"Recommendations for {manuscript_id}")
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Journal")
        table.add_column("Score", justify="right")
        table.add_column("Policy")
        table.add_column("History")
        for r in results:
            policy_str = "[green]pass[/green]" if r.policy.passed else f"[red]fail[/red]: {r.policy.reason}"
            hist = r.user_history
            hist_str = f"acc:{hist.get('prior_acceptances',0)} rej:{hist.get('prior_rejections',0)}"
            table.add_row(
                str(r.rank) if r.policy.passed else "—",
                r.journal_name[:45],
                f"{r.score:.3f}" if r.policy.passed else "—",
                policy_str,
                hist_str,
            )
            if explain and r.policy.passed:
                d = r.decomposition
                console.print(
                    f"     [dim]abstract:{d.dense_abstract:.3f}  "
                    f"lexical:{d.lexical_techniques:.3f}  "
                    f"claim:{d.principal_claim:.3f}  "
                    f"dispersion:{d.topk_dispersion_penalty:.3f}[/dim]"
                )
                if r.rationale:
                    console.print(f"     [italic]{r.rationale}[/italic]")
                for art in r.nearest_articles[:3]:
                    doi = art.get("doi", "")
                    title = (art.get("title") or "")[:70]
                    sim = art.get("similarity", 0)
                    console.print(f"     [dim]  {sim:.3f}  {title}  doi:{doi}[/dim]")
        console.print(table)
    else:
        print(json.dumps([r.to_dict() for r in results], indent=2))


# ── submission ────────────────────────────────────────────────────────────────

@submission_app.command("create")
def submission_create(
    manuscript_id: str = typer.Option(..., "--manuscript", "-m"),
    journal_id: str = typer.Option(..., "--journal", "-j"),
):
    """Create a new submission record in 'drafting' state."""
    from jfr.tracker import create_submission
    conn = _get_conn()
    sub_id = create_submission(conn, manuscript_id, journal_id)
    rprint(f"[green]✓[/green] Created [bold]{sub_id}[/bold] in state [bold]drafting[/bold]")


@submission_app.command("transition")
def submission_transition(
    sub_id: str = typer.Argument(...),
    to_state: str = typer.Argument(...),
    note: Optional[str] = typer.Option(None, "--note"),
):
    """Transition a submission to a new state."""
    from jfr.tracker import transition_submission, get_submission, InvalidTransitionError
    conn = _get_conn()
    sub = get_submission(conn, sub_id)
    if not sub:
        rprint(f"[red]Submission {sub_id!r} not found[/red]")
        raise typer.Exit(1)
    from_state = sub["current_state"]
    try:
        transition_submission(conn, sub_id, to_state, notes=note)
        rprint(f"[green]✓[/green] {sub_id}: [bold]{from_state}[/bold] → [bold]{to_state}[/bold]")
    except InvalidTransitionError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@submission_app.command("show")
def submission_show(sub_id: str):
    """Show full submission record."""
    from jfr.tracker import get_submission
    conn = _get_conn()
    sub = get_submission(conn, sub_id)
    if not sub:
        rprint(f"[red]Submission {sub_id!r} not found[/red]")
        raise typer.Exit(1)
    print(json.dumps(sub, indent=2))


@submission_app.command("comment")
def submission_comment(
    sub_id: str = typer.Argument(...),
    reviewer: int = typer.Option(..., "--reviewer", "-r"),
    round_: int = typer.Option(1, "--round"),
    text: str = typer.Option(..., "--text"),
):
    """Add a reviewer comment to a submission."""
    from jfr.tracker import add_reviewer_comment
    conn = _get_conn()
    add_reviewer_comment(conn, sub_id, reviewer, text, round=round_)
    rprint(f"[green]✓[/green] Comment added for reviewer {reviewer} (round {round_})")


# ── rag: search ─────────────────────────────────────

@rag_search_app.command("query")
def rag_search_query(
    query: str = typer.Argument(..., help="Search query for RAG papers"),
    limit: int = typer.Option(10, "-n", "--limit", help="Number of results"),
):
    """Search your paper collection via RAG."""
    import urllib.request, urllib.parse, json
    try:
        url = f"http://127.0.0.1:8765/api/rag/search?query={urllib.parse.quote(query)}&limit={limit}"
        req = urllib.request.urlopen(url, timeout=15)
        data = json.loads(req.read())
    except Exception as e:
        rprint(f"[red]Error:[/red] Could not reach web UI at http://127.0.0.1:8765 — run 'jfr web serve' first. ({e})")
        raise typer.Exit(1)

    results = data.get("results", [])
    rag_info = data.get("rag_info", {})

    rprint(f"\n[cyan]\u2192[/cyan] Found {data.get('total', len(results))} results for: [bold]{query}[/bold]")
    if rag_info:
        rprint(f"[dim]Index: {rag_info.get('papers_indexed', '?')} papers indexed[/dim]\n")

    if not results:
        rprint("[yellow]No results found[/yellow]")
        return

    if _is_tty():
        table = Table(title="Search Results")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Authors", style="dim")
        table.add_column("Year", style="dim")
        table.add_column("Score", justify="right", style="dim")

        for i, result in enumerate(results[:limit], 1):
            title = result.get("title", "N/A")
            authors = result.get("authors", "N/A")
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors[:3])
            year = result.get("year", "N/A")
            score = result.get("match_score", "?")

            table.add_row(
                str(i),
                title[:60] if title else "Untitled",
                str(authors)[:45],
                str(year),
                str(score),
            )
        console.print(table)
    else:
        print(json.dumps(data, indent=2))


# ── rag: link ─────────────────────────────────────────

@rag_link_app.command("add")
def rag_link_add(
    manuscript_id: str = typer.Argument(..., help="Manuscript ID"),
    paper_id: str = typer.Argument(..., help="RAG paper ID to link"),
    link_type: str = typer.Option("related", "-t", "--type", help="Link type: cites, supports, contrasts, background, extends"),
    note: str = typer.Option("", "-n", "--note", help="Optional note about the link"),
):
    """Link a RAG paper to a manuscript."""
    import urllib.request, urllib.parse, json, http.client, ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPConnection("127.0.0.1", port=8765, timeout=10)
        body = f"ms_id={urllib.parse.quote(manuscript_id)}&paper_id={urllib.parse.quote(paper_id)}&link_type={urllib.parse.quote(link_type)}&note={urllib.parse.quote(note)}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        conn.request("POST", "/api/rag/links", body, headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        if isinstance(data, dict) and "success" in data and data["success"]:
            rprint(f"[green]\u2713[/green] Linked [bold]{paper_id}[/bold] to [bold]{manuscript_id}[/bold] as '{link_type}'")
            if note:
                rprint(f"     Note: {note}")
        else:
            rprint(f"[red]Error:[/red] {data.get('error', 'Link creation failed')}")
            raise typer.Exit(1)
    except urllib.error.URLError as e:
        rprint(f"[red]Error:[/red] Could not reach web UI — run 'jfr web serve' first. ({e})")
        raise typer.Exit(1)


@rag_link_app.command("list")
def rag_link_list(ms_id: str = typer.Argument(..., help="Manuscript ID to list papers for")):
    """List all RAG papers linked to a manuscript."""
    import urllib.request, json
    try:
        url = f"http://127.0.0.1:8765/api/rag/links/{ms_id}"
        resp = json.loads(urllib.request.urlopen(url, timeout=15).read())
        links = resp.get("links", [])
    except Exception as e:
        rprint(f"[red]Error:[/red] Could not connect ({e})")
        raise typer.Exit(1)

    if not links:
        rprint(f"[yellow]No linked papers for {ms_id}[/yellow]")
        return

    rprint(f"\n[cyan]\u2192[/cyan] Linked papers for [bold]{ms_id}[/bold]:")

    if _is_tty():
        table = Table(title=f"Paper Links for {ms_id}")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Type", style="dim")
        table.add_column("Note", style="dim")

        for link in links:
            title = link.get("title", link.get("paper_title", link.get("rag_paper_id", "?")))[:50]
            table.add_row(
                str(link.get("rag_paper_id", link.get("id", "?"))),
                title,
                link.get("link_type", "?"),
                (link.get("note") or "")[:30],
            )
        console.print(table)
    else:
        print(json.dumps(links, indent=2))


# ── rag: lab ─────────────────────────────────────────

@rag_lab_app.command("stats")
def rag_lab_stats():
    """Show research lab statistics."""
    import urllib.request, json
    try:
        resp = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/rag/lab", timeout=15).read())
    except Exception as e:
        rprint(f"[red]Error:[/red] Could not connect ({e})")
        raise typer.Exit(1)

    rprint("\n[bold]Research Lab Statistics[/bold]\n")

    if _is_tty():
        table = Table()
        table.add_column("Category", style="cyan")
        table.add_column("Value", justify="right", style="white")

        table.add_row("\U0001f4dd Manuscripts", str(resp.get("manuscripts", 0)))
        table.add_row("\U0001f4ec Active Submissions", str(resp.get("active_submissions", 0)))
        table.add_row("\U0001f4da Papers Indexed", str(resp.get("rag", {}).get("papers_indexed", 0)))
        table.add_row("\U0001f4c4 Chunks Indexed", str(resp.get("rag", {}).get("chunks_indexed", 0)))
        table.add_row("\U0001f9e0 Memories", str(resp.get("rag", {}).get("total_memories", 0)))
        table.add_row("\U0001f578\xef\xb8\x8f  Entities", str(resp.get("graph", {}).get("entities", 0)))

        console.print(table)
    else:
        print(json.dumps(resp, indent=2))
    rprint()


@rag_lab_app.command("open")
def rag_lab_open(ms_id: str = typer.Argument(None, help="Manuscript to open")):
    """Open the research lab in a browser."""
    import webbrowser
    url = "http://127.0.0.1:8765/lab" + (f"?ms={ms_id}" if ms_id else "")
    rprint(f"[cyan]\u2192[/cyan] Opening Research Lab at [bold]http://127.0.0.1:8765/lab[/bold]")
    try:
        webbrowser.open(url)
    except Exception as e:
        rprint(f"[yellow]   Error auto-opening: {e}[/yellow]")
        rprint(f"[dim]   Open manually: {url}[/dim]")


# ── view ──────────────────────────────────────────────────────────────────────

@view_app.command("active")
def view_active():
    """Show all active (non-terminal) submissions sorted by elapsed time."""
    from jfr.tracker import list_active_submissions
    from datetime import datetime, timezone
    conn = _get_conn()
    subs = list_active_submissions(conn)
    if _is_tty():
        table = Table(title="Active Submissions")
        table.add_column("Sub ID", style="cyan")
        table.add_column("Manuscript")
        table.add_column("Journal")
        table.add_column("State")
        table.add_column("Days", justify="right")
        now = datetime.now(timezone.utc)
        for s in subs:
            upd = datetime.fromisoformat(s["updated_at"].replace("Z", "+00:00"))
            days = (now - upd).days
            table.add_row(
                s["id"], s["manuscript_title"][:40],
                s["journal_name"][:30], s["current_state"], str(days),
            )
        console.print(table)
    else:
        print(json.dumps(subs, indent=2))


@view_app.command("manuscript")
def view_manuscript(manuscript_id: str):
    """Show all submission attempts for a manuscript."""
    from jfr.tracker import list_submissions_for_manuscript
    conn = _get_conn()
    rows = list_submissions_for_manuscript(conn, manuscript_id)
    print(json.dumps(rows, indent=2))


@view_app.command("stalled")
def view_stalled(
    days: int = typer.Option(90, "--days", help="Alert threshold in days"),
    deliver: str = typer.Option("", "--deliver", help="Delivery target: whatsapp, local, or origin"),
    thresholds: str = typer.Option("", "--thresholds", help="Per-state thresholds: under_review=75,revision_requested=14,awaiting_approval=60"),
):
    """List active submissions that have not advanced beyond the threshold."""
    from jfr.tracker import list_active_submissions
    from datetime import datetime, timezone
    conn = _get_conn()
    subs = list_active_submissions(conn)
    now = datetime.now(timezone.utc)

    # Parse per-state thresholds
    state_thresholds = {}
    if thresholds:
        for pair in thresholds.split(","):
            state, days_val = pair.split("=")
            state_thresholds[state.strip()] = int(days_val.strip())

    stalled = []
    for s in subs:
        ref = s.get("submitted_at") or s.get("updated_at") or s.get("created_at")
        if not ref:
            continue
        dt = datetime.fromisoformat(ref.replace("Z", "+00:00"))
        elapsed = (now - dt).days
        # Use per-state threshold if available, else default
        thresh = state_thresholds.get(s["current_state"], days)
        if elapsed >= thresh:
            stalled.append({**dict(s), "days_elapsed": elapsed, "threshold": thresh})

    if _is_tty():
        if not stalled:
            rprint(f"[green]✓[/green] No submissions stalled beyond configured thresholds.")
            return
        table = Table(title=f"Stalled Submissions")
        table.add_column("Sub ID", style="cyan")
        table.add_column("Manuscript")
        table.add_column("Journal")
        table.add_column("State")
        table.add_column("Days", justify="right", style="bold red")
        table.add_column("Threshold", justify="right")
        for s in sorted(stalled, key=lambda x: -x["days_elapsed"]):
            table.add_row(
                s["id"], s["manuscript_title"][:40],
                s["journal_name"][:30], s["current_state"],
                str(s["days_elapsed"]), str(s["threshold"]),
            )
        console.print(table)
    else:
        import json as _json
        print(_json.dumps(stalled, indent=2))

    if deliver == "whatsapp":
        if not stalled:
            rprint("[green]✓[/green] No stalled submissions — nothing to deliver.")
            return
        msg_lines = [
            "⚠ JFR Stalled Submissions Alert",
            "",
        ]
        for s in sorted(stalled, key=lambda x: -x["days_elapsed"]):
            msg_lines.append(
                f"  {s['id']}: {s['manuscript_title'][:40]} → {s['journal_name']} "
                f"[{s['current_state']}] {s['days_elapsed']}d (threshold: {s['threshold']}d)"
            )
        msg_lines.append("")
        msg_lines.append("Consider sending a status inquiry email to the editor.")
        whatsapp_msg = "\n".join(msg_lines)
        try:
            from hermes_tools import send_message
            send_message(action="send", target="whatsapp", message=whatsapp_msg)
            rprint(f"[green]✓[/green] Stalled alerts delivered to WhatsApp")
        except ImportError:
            rprint(whatsapp_msg)
            rprint("[dim](push delivery not configured — provide a `hermes_tools` module to enable)[/dim]")


@view_app.command("journal")
def view_journal(journal_id: str):
    """Show submission history for a journal."""
    from jfr.tracker import list_submissions_for_journal
    conn = _get_conn()
    rows = list_submissions_for_journal(conn, journal_id)
    print(json.dumps(rows, indent=2))


# ── web ───────────────────────────────────────────────────────────────────────

web_app = typer.Typer(help="Web interface", no_args_is_help=True)
app.add_typer(web_app, name="web")


@web_app.command("serve")
def web_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8765, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development)"),
):
    """Start the web interface on localhost:8765."""
    import uvicorn
    s = get_settings()
    s.ensure_dirs()
    rprint(f"[cyan]→[/cyan] Starting jfr web UI at http://{host}:{port}")
    uvicorn.run("jfr.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
