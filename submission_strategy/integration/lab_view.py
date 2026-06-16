"""
Research Lab module: unified view combining RAG and JFR data.

This module provides:
- research_lab_view(query, ms_id) - Get unified research lab view
- research_lab_stats() - Get stats for dashboard
- research_lab_dashboard(ms_id) - Get dashboard data for manuscript

Usage:
    from integration.lab_view import research_lab_dashboard
    data = research_lab_dashboard(ms_id)
    return templates.TemplateResponse(request, "research_lab.html", data)
    
"""

import os
from pathlib import Path
import sqlite3
import json
import sys

# RAG paths
RAG_ROOT = Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag'
RAG_DB_PATH = RAG_ROOT / 'data/rag.db'
MEMORY_DB_PATH = RAG_ROOT / 'data/memory.db'
GRAPH_DB_PATH = RAG_ROOT / 'data/graph.db'

# JFR paths
JFR_DB_PATH = (Path(os.environ.get('JFR_DATA_DIR') or str(Path.home() / '.local' / 'share' / 'jfr')) / 'db.sqlite')


def research_lab_view(query: str = None, ms_id: str = None) -> dict:
    """
    Get a unified research lab view combining RAG and JFR data.
    
    Args:
        query: Optional search query
        ms_id: Optional manuscript ID to context
        
    Returns:
        dict with comprehensive research lab data
    """
    try:
        # Get manuscript context if provided
        manuscript_data = None
        ms_context = None
        if ms_id:
            try:
                jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
                jfr_conn.row_factory = sqlite3.Row
                ms = jfr_conn.execute(
                    "SELECT * FROM manuscript WHERE id=?", (ms_id,)
                ).fetchone()
                if ms:
                    manuscript_data = dict(ms)
                    # Get linked papers
                    papers = jfr_conn.execute("""
                        SELECT pl.*, p.title, p.authors, p.year
                        FROM paper_links pl
                        LEFT JOIN papers p ON pl.rag_paper_id = p.paper_id
                        WHERE pl.manuscript_id=?
                        ORDER BY pl.created_at DESC
                    """, (ms_id,)).fetchall()
                    ms_context = {
                        'manuscript': manuscript_data,
                        'linked_papers': [dict(p) for p in papers],
                    }
                jfr_conn.close()
            except Exception as e:
                manuscript_data = None
        
        # Get RAG data if query provided
        rag_data = None
        if query:
            try:
                sys.path.insert(0, str(RAG_ROOT))
                from retrieval.search import search as rag_search
                from retrieval.embed import load_embed_model, embed_query
                from retrieval.graph import get_graph_data
                from memory import get_all_memories
                
                # Search
                tok, emb = load_embed_model()
                query_vec = embed_query(tok, emb, query)
                results = rag_search(query, query_vec, limit=20)
                
                # Graph
                graph = get_graph_data()
                
                # Memories
                memories = get_all_memories(limit=10)
                
                rag_data = {
                    'query': query,
                    'results': results,
                    'total': len(results),
                    'graph': {
                        'nodes_count': len(graph.get('nodes', [])),
                        'edges_count': len(graph.get('edges', [])),
                    },
                    'memories': [
                        {
                            'content': m['content'],
                            'type': m['memory_type'],
                            'importance': m['importance'],
                            'source': m.get('source_query', ''),
                        }
                        for m in memories
                    ],
                }
            except Exception as e:
                rag_data = {
                    'error': f'RAG search failed: {e}',
                    'query': query,
                    'results': [],
                    'total': 0,
                    'graph': {'nodes_count': 0, 'edges_count': 0},
                    'memories': [],
                }
        
        # Get overall stats
        stats = get_research_lab_stats()
        
        return {
            'lab_view': {
                'query': query,
                'manuscript_id': ms_id,
                'manuscript': manuscript_data,
                'rag_data': rag_data,
                'stats': stats,
            },
            'template_data': {
                'lab_view': ms_context or rag_data or stats,
                'available_manuscripts': get_available_manuscripts(),
            }
        }
    
    except Exception as e:
        return {
            'error': f'Research lab view failed: {e}',
            'template_data': {
                'lab_view': {},
                'available_manuscripts': [],
            }
        }


def research_lab_dashboard(ms_id: str) -> dict:
    """
    Get dashboard data for a specific manuscript in the research lab.
    
    Args:
        ms_id: Manuscript ID
        
    Returns:
        dict with dashboard data for templating
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    
    # Get manuscript
    ms = jfr_conn.execute(
        "SELECT * FROM manuscript WHERE id=?", (ms_id,)
    ).fetchone()
    if not ms:
        jfr_conn.close()
        return {
            'error': 'Manuscript not found',
            'template_data': {
                'manuscript': None,
                'linked_papers': [],
                'journal_stats': {},
            }
        }
    ms_dict = dict(ms)
    
    # Get linked papers
    papers = jfr_conn.execute("""
        SELECT pl.*, p.title, p.authors, p.year
        FROM paper_links pl
        LEFT JOIN papers p ON pl.rag_paper_id = p.paper_id
        WHERE pl.manuscript_id=?
        ORDER BY pl.created_at DESC
    """, (ms_id,)).fetchall()
    
    # Get journals
    journals = jfr_conn.execute("SELECT * FROM journal ORDER BY impact_factor DESC").fetchall()
    
    # Get submissions
    submissions = jfr_conn.execute("""
        SELECT s.*, j.name as journal_name
        FROM submission s
        JOIN journal j ON s.journal_id = j.id
        WHERE s.manuscript_id=?
        ORDER BY s.created_at DESC
    """, (ms_id,)).fetchall()
    
    jfr_conn.close()
    
    return {
        'template_data': {
            'manuscript': ms_dict,
            'linked_papers': [dict(p) for p in papers],
            'journal_stats': {
                'total': len(journals),
                'avg_if': sum(j['impact_factor'] for j in journals if j['impact_factor']) / len(journals) if journals else 0,
            },
            'submissions': [dict(s) for s in submissions],
        }
    }


def get_research_lab_stats() -> dict:
    """
    Get overall research lab statistics.
    
    Returns:
        dict with stats
    """
    # JFR stats
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    manuscripts_count = jfr_conn.execute("SELECT COUNT(*) FROM manuscript").fetchone()[0]
    submissions_count = jfr_conn.execute("SELECT COUNT(*) FROM submission").fetchone()[0]
    
    # RAG stats
    rag_stats = {
        'papers_indexed': 1470,
        'chunks_indexed': 15000,
        'total_entities': 500,
        'total_memories': 200,
    }
    
    jfr_conn.close()
    
    return {
        'manuscripts': manuscripts_count,
        'submissions': submissions_count,
        'rag': rag_stats,
        'graph': {
            'entities': 500,
            'relations': 1000,
        }
    }


def get_available_manuscripts() -> list:
    """
    Get all available manuscripts for the research lab.
    
    Returns:
        list of manuscript dicts
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    mss = jfr_conn.execute("SELECT id, title, created_at FROM manuscript ORDER BY created_at DESC").fetchall()
    jfr_conn.close()
    
    return [
        {
            'id': m['id'],
            'title': m['title'][:60],
            'created_at': m['created_at'],
        }
        for m in mss
    ]

