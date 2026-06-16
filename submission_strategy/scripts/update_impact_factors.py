#!/usr/bin/env python3
"""Update impact factors for all journals via OpenAlex API.

OpenAlex provides impact_factor data per source (journal).
Fetches current IFs and updates both journals.yaml and the SQLite DB.
"""

import json
import sqlite3
from pathlib import Path

OPENALEX_API = "https://api.openalex.org/works"
UA = "jfr/0.1 (mailto:contact@example.com)"

import httpx

JOURNALS_YAML = Path(__file__).parent.parent / "data" / "journals.yaml"
DB_PATH = Path.home() / ".local" / "share" / "jfr" / "db.sqlite"


def fetch_impact_factor(issn: str) -> float | None:
    """Fetch impact factor for a journal via OpenAlex source API.
    
    Uses summary_stats.2yr_mean_citedness as impact factor proxy.
    """
    with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
        resp = client.get(
            f"https://api.openalex.org/sources?filter=issn:{issn}",
            params={"per-page": 1},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        source = results[0]
        summary = source.get("summary_stats", {})
        return summary.get("2yr_mean_citedness")


def update_yaml(journals_yaml: Path, new_ifs: dict[str, float]) -> None:
    """Update impact_factor in journals.yaml for each journal."""
    content = journals_yaml.read_text()
    lines = content.split("\n")
    updated = 0
    for i, line in enumerate(lines):
        for jid, new_if in new_ifs.items():
            # Find the journal block and its impact_factor line
            if f"- id: {jid}" in line:
                # Look ahead for impact_factor line in this journal's block
                for j in range(i + 1, min(i + 20, len(lines))):
                    if "impact_factor:" in lines[j]:
                        old_if = float(lines[j].split("impact_factor:")[1].strip())
                        lines[j] = f"    impact_factor: {new_if}"
                        updated += 1
                        break
                break
    journals_yaml.write_text("\n".join(lines))
    print(f"Updated {updated} impact factors in journals.yaml")


def update_db(db_path: Path, new_ifs: dict[str, float]) -> None:
    """Update impact_factor in SQLite journal table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    updated = 0
    for jid, new_if in new_ifs.items():
        conn.execute("UPDATE journal SET impact_factor=? WHERE id=?", (new_if, jid))
        updated += conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    print(f"Updated {updated} impact factors in SQLite DB")


def main():
    # Load current journal list from YAML
    content = JOURNALS_YAML.read_text()
    journals = {}
    current_lines = content.split("\n")
    for i, line in enumerate(current_lines):
        if "id:" in line and "- id:" in line:
            jid = line.split("id:")[1].strip()
            # Look ahead for ISSN
            for j in range(i + 1, min(i + 15, len(current_lines))):
                if "issn_electronic:" in current_lines[j]:
                    issn = current_lines[j].split("issn_electronic:")[1].strip().strip('"')
                    journals[jid] = issn
                    break

    print(f"Found {len(journals)} journals to update")

    # Fetch impact factors with delays to avoid rate limiting
    new_ifs = {}
    import time as _time
    for jid, issn in journals.items():
        print(f"  Fetching IF for {jid} ({issn})...")
        if_val = fetch_impact_factor(issn)
        if if_val is not None:
            new_ifs[jid] = if_val
            print(f"    New IF: {if_val}")
        else:
            print(f"    ⚠ No IF available")
        _time.sleep(0.5)  # Rate limit protection

    # Update YAML
    update_yaml(JOURNALS_YAML, new_ifs)

    # Update DB
    update_db(DB_PATH, new_ifs)

    # Print summary
    print("\n=== Impact Factor Summary ===")
    for jid, new_if in sorted(new_ifs.items(), key=lambda x: -x[1]):
        print(f"  {jid}: {new_if}")


if __name__ == "__main__":
    main()
