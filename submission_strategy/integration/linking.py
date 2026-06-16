"""
Linking module: connects manuscripts to RAG papers persistently.

This module provides functions for:
- Linking manuscripts to RAG papers
- Getting linked papers for a manuscript
- Managing paper-manuscript relationships

Usage:
    from integration.linking import link_manuscript_to_paper
    from integration.linking import get_linked_papers
    from integration.linking import get_manuscript_paper_links
    
"""

import os
from pathlib import Path
import sqlite3
import json

# RAG paths
RAG_DB_PATH = (Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag' / 'data' / 'rag.db')

# JFR paths
JFR_DB_PATH = (Path(os.environ.get('JFR_DATA_DIR') or str(Path.home() / '.local' / 'share' / 'jfr')) / 'db.sqlite')

VALID_LINK_TYPES = {
    'cites',         # Manuscript cites this paper
    'contrasts',     # Manuscript contrasts with this paper
    'supports',      # Manuscript is supported by this paper
    'background',    # Background for this paper
    'extends',       # Manuscript extends this paper's work
    'related',       # Related to this paper
}

def link_manuscript_to_paper(
    manuscript_id: str,
    paper_id: str,
    link_type: str = 'related',
    note: str = ''
) -> bool:
    """
    Link a manuscript to a RAG paper.
    
    Args:
        manuscript_id: Manuscript ID in JFR DB
        paper_id: Paper ID in RAG DB
        link_type: Type of relationship (cites, contrasts, supports, etc.)
        note: Optional note about the link
        
    Returns:
        True if successful, False if not
    """
    if link_type not in VALID_LINK_TYPES:
        raise ValueError(f"Invalid link_type: {link_type}. Must be one of {VALID_LINK_TYPES}")
    
    # Check manuscript exists
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    ms = jfr_conn.execute(
        "SELECT id FROM manuscript WHERE id=?", (manuscript_id,)
    ).fetchone()
    if not ms:
        jfr_conn.close()
        raise ValueError(f"Manuscript {manuscript_id} not found")
    
    # Check paper exists
    rag_conn = sqlite3.connect(str(RAG_DB_PATH))
    paper = rag_conn.execute(
        "SELECT * FROM papers WHERE paper_id=?", (paper_id,)
    ).fetchone()
    if not paper:
        rag_conn.close()
        raise ValueError(f"Paper {paper_id} not found in RAG")
    
    # Insert link
    jfr_conn.execute("""
        INSERT INTO paper_links (manuscript_id, rag_paper_id, link_type, note)
        VALUES (?, ?, ?, ?)
    """, (manuscript_id, paper_id, link_type, note))
    jfr_conn.commit()
    jfr_conn.close()
    rag_conn.close()
    
    return True


def get_linked_papers(manuscript_id: str) -> list:
    """
    Get all RAG papers linked to a manuscript.
    
    Args:
        manuscript_id: Manuscript ID in JFR DB
        
    Returns:
        List of dicts with paper details
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    jfr_conn.row_factory = sqlite3.Row

    links = jfr_conn.execute(
        "SELECT * FROM paper_links WHERE manuscript_id=? ORDER BY created_at DESC",
        (manuscript_id,),
    ).fetchall()
    jfr_conn.close()

    if not links:
        return []

    # Enrich with paper metadata from rag.db
    rag_conn = sqlite3.connect(str(RAG_DB_PATH))
    rag_conn.row_factory = sqlite3.Row
    paper_ids = [link['rag_paper_id'] for link in links]
    placeholders = ','.join('?' * len(paper_ids))
    paper_rows = rag_conn.execute(
        f"SELECT paper_id, title, authors, year, source_pdf FROM papers "
        f"WHERE paper_id IN ({placeholders})",
        paper_ids,
    ).fetchall()
    rag_conn.close()
    by_id = {p['paper_id']: dict(p) for p in paper_rows}

    return [
        {
            'id': link['id'],
            'rag_paper_id': link['rag_paper_id'],
            'link_type': link['link_type'],
            'note': link['note'],
            'created_at': link['created_at'],
            'title':     by_id.get(link['rag_paper_id'], {}).get('title', link['rag_paper_id']),
            'authors':   by_id.get(link['rag_paper_id'], {}).get('authors', ''),
            'year':      by_id.get(link['rag_paper_id'], {}).get('year'),
            'source_pdf':by_id.get(link['rag_paper_id'], {}).get('source_pdf', ''),
        }
        for link in links
    ]


def get_manuscript_paper_links(manuscript_id: str) -> dict:
    """
    Get all paper links for a manuscript with metadata.
    
    Args:
        manuscript_id: Manuscript ID in JFR DB
        
    Returns:
        dict with manuscript and linked papers
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    
    # Get manuscript
    ms = jfr_conn.execute(
        "SELECT * FROM manuscript WHERE id=?", (manuscript_id,)
    ).fetchone()
    if not ms:
        jfr_conn.close()
        return {'error': 'Manuscript not found'}
    
    # Get linked papers
    links = get_linked_papers(manuscript_id)
    
    # Get paper counts by type
    type_counts = {}
    for link in links:
        t = link['link_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    result = {
        'manuscript': dict(ms),
        'linked_papers': links,
        'link_stats': {
            'total': len(links),
            'by_type': type_counts,
        },
    }
    
    jfr_conn.close()
    return result


def delete_link(link_id: int) -> bool:
    """
    Delete a paper link.
    
    Args:
        link_id: Link ID in paper_links table
        
    Returns:
        True if successful
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    jfr_conn.execute("DELETE FROM paper_links WHERE id=?", (link_id,))
    jfr_conn.commit()
    jfr_conn.close()
    return True
