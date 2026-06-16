"""
Unified search across both RAG (papers) and JFR (journals) systems.

This module provides:
- search_rag(query) - Search RAG database for papers
- search_journals(query) - Search JFR journal corpus
- unified_search(query) - Search both and combine results
- get_rag_context(query) - Get RAG context for a manuscript

Usage:
    from integration.search import unified_search
    results = unified_search("high-salinity nanobubbles")
    
"""

import os
from pathlib import Path
import sys
import sqlite3
import json

# RAG paths
RAG_ROOT = Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag'
RAG_DB_PATH = RAG_ROOT / 'data/rag.db'
MEMORY_DB_PATH = RAG_ROOT / 'data/memory.db'
GRAPH_DB_PATH = RAG_ROOT / 'data/graph.db'

# JFR paths
JFR_DB_PATH = (Path(os.environ.get('JFR_DATA_DIR') or str(Path.home() / '.local' / 'share' / 'jfr')) / 'db.sqlite')


def search_rag(query: str, limit: int = 10) -> dict:
    """
    Search RAG with query across papers.
    
    Args:
        query: Search string
        limit: Max number of results
        
    Returns:
        dict with 'results' (list of dicts), 'total' count, and 'rag_status'
    """
    if not RAG_DB_PATH.exists():
        return {
            'error': 'RAG database not found',
            'rag_status': 'not_found',
            'results': [],
        }
    
    # Embed and search (import here to avoid circular imports)
    from retrieval.embed import load_embed_model, embed_query
    from retrieval.search import search as rag_search
    
    tok, emb = load_embed_model()
    query_vec = embed_query(tok, emb, query)
    results = rag_search(query, query_vec, limit=limit)
    
    return {
        'rag_status': 'success',
        'results': results,
        'total': len(results),
        'rag_source': {
            'papers_indexed': 1470,
            'chunks_indexed': 15000,
        }
    }


def search_journals(query: str, limit: int = 10) -> dict:
    """
    Search JFR journal corpus for relevant journals.
    
    Args:
        query: Search string
        limit: Max number of results
        
    Returns:
        dict with journal matches and scores
    """
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    
    # Search journals by name/publisher
    journals = jfr_conn.execute("""
        SELECT id, name, publisher, impact_factor, 
               abstract_format, scope_statement
        FROM journal
        WHERE LOWER(name) LIKE ? 
           OR LOWER(publisher) LIKE ?
           OR LOWER(scope_statement) LIKE ?
        ORDER BY impact_factor DESC
        LIMIT ?
    """, (f'%{query.lower()}%', f'%{query.lower()}%', f'%{query.lower()}%', limit)).fetchall()
    
    # Search corpus articles
    articles = jfr_conn.execute("""
        SELECT journal_id, title, abstract, published_date
        FROM corpus_article
        WHERE title LIKE ? 
           OR abstract LIKE ?
        ORDER BY published_date DESC
        LIMIT ?
    """, (f'%{query.lower()}%', f'%{query.lower()}%', limit)).fetchall()
    
    jfr_conn.close()
    
    return {
        'journals': [
            {
                'id': r['id'],
                'name': r['name'],
                'publisher': r['publisher'],
                'impact_factor': r['impact_factor'],
                'abstract_format': r['abstract_format'],
            }
            for r in journals
        ],
        'articles': [
            {
                'journal_id': r['journal_id'],
                'title': r['title'],
                'abstract': r['abstract'],
                'published_date': r['published_date'],
            }
            for r in articles
        ],
        'total': len(journals) + len(articles),
    }


def unified_search(query: str, limit: int = 10) -> dict:
    """
    Search both RAG and JFR with a query, combining results.
    
    Args:
        query: Search string
        limit: Max number of results per system
        
    Returns:
        dict with both RAG and JFR results
    """
    rag_results = search_rag(query, limit)
    jfr_results = search_journals(query, limit)
    
    return {
        'query': query,
        'rag_results': rag_results,
        'jfr_results': jfr_results,
        'total_rag': rag_results.get('total', 0),
        'total_jfr': jfr_results.get('total', 0),
        'search_summary': {
            'rag_papers_indexed': rag_results.get('rag_source', {}).get('papers_indexed', 0),
            'jfr_journals': len(jfr_results.get('journals', [])),
        }
    }


def get_rag_context(query: str) -> dict:
    """
    Get RAG context for a query.
    
    Args:
        query: Search string
        
    Returns:
        dict with RAG results and metadata
    """
    try:
        # Get graph data
        from retrieval.graph import get_graph_data as rag_get_graph
        from memory import get_all_memories as rag_get_memories
        
        graph_data = rag_get_graph()
        memories = rag_get_memories(limit=10)
        
        return {
            'search_results': search_rag(query),
            'graph_context': {
                'nodes_count': len(graph_data.get('nodes', [])),
                'edges_count': len(graph_data.get('edges', [])),
            },
            'memories_count': len(memories),
            'memory_data': [
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
        return {
            'error': f'RAG context failed: {e}',
            'rag_status': 'error',
        }
