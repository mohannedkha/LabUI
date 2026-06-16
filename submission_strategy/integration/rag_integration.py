"""
Unified search across RAG (papers) and JFR (journals) systems.

Searches your literature collection AND journal corpus simultaneously.

Usage:
    from integration.search import unified_search
    results = unified_search("high-salinity nanobubbles")
    
"""

import os
from pathlib import Path
import sqlite3
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag'))
import config as rag_config
from retrieval.embed import load_embed_model, embed_query as rag_embed_query
from retrieval.search import search as rag_search
import config as rag_config
from retrieval.embed import embed_query as rag_embed_query

# RAG paths
RAG_DB_PATH = (Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag' / 'data' / 'rag.db')
MEMORY_DB_PATH = (Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag' / 'data' / 'memory.db')
GRAPH_DB_PATH = (Path(__file__).resolve().parents[2] / 'Local_Rag' / 'rag' / 'data' / 'graph.db')

# JFR paths
JFR_DB_PATH = (Path(os.environ.get('JFR_DATA_DIR') or str(Path.home() / '.local' / 'share' / 'jfr')) / 'db.sqlite')


def get_manuscript_context(ms_id: str) -> dict:
    """Get manuscript + RAG related papers for a manuscript
    
    Returns dict with:
        manuscript: dict
        rag_papers: list of dicts (from rag.db)
        graph_entities: list of dicts (from graph.db)
        related_papers: list of dicts (linked via paper_links)
        memories: list of dicts (from memory.db)
    """
    # Get manuscript from JFR
    jfr_conn = sqlite3.connect(str(JFR_DB_PATH))
    jfr_conn.row_factory = sqlite3.Row
    ms_row = jfr_conn.execute("SELECT * FROM manuscript WHERE id=?", (ms_id,)).fetchone()
    if not ms_row:
        return {'error': 'No manuscript found', 'manuscript': None}
    ms = dict(ms_row)
    
    # Get linked papers from paper_links
    paper_links = [
        dict(r) for r in jfr_conn.execute(
            "SELECT * FROM paper_links WHERE manuscript_id=?", (ms_id,)
        ).fetchall()
    ]
    
    # Get RAG papers via graph for manuscript keywords
    keywords = []
    if 'techniques_json' in ms and ms['techniques_json']:
        try:
            keywords.extend(json.loads(ms['techniques_json']))
        except (json.JSONDecodeError, TypeError):
            pass
    if 'abstract' in ms:
        keywords.append(ms['abstract'][:200])
    
    rag_results = []
    if keywords:
        # Search RAG graph for related entities
        from retrieval.graph import search_entities_fts as rag_search_entities
        entities = rag_search_entities(str(keywords[0]), limit=20)
        
        # Get graph data
        from retrieval.graph import get_graph_data as rag_get_graph
        graph_data = rag_get_graph()
        
        # Search memories for related content
        from memory import get_all_memories as rag_get_memories
        memories = rag_get_memories(limit=10)
        
        # Get relevant papers from rag.db
        rag_conn = sqlite3.connect(str(RAG_DB_PATH))
        rag_conn.row_factory = sqlite3.Row
        # Search chunks for keywords
        query_chunks = rag_conn.execute(
            "SELECT c.*, p.title, p.authors, p.year "
            "FROM chunks c JOIN papers p ON c.paper_id = p.paper_id "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT 10", (str(keywords[0]),)
        ).fetchall()
        
        rag_papers = []
        seen_paper_ids = set()
        for chunk in query_chunks:
            c = dict(chunk)
            if c['paper_id'] not in seen_paper_ids:
                rag_papers.append({
                    'paper_id': c['paper_id'],
                    'title': c['title'],
                    'authors': c['authors'],
                    'year': c['year'],
                    'section': c['section_name'],
                    'page': f"{c['page_start']}-{c['page_end']}",
                })
                seen_paper_ids.add(c['paper_id'])
        
        rag_conn.close()
        rag_results.append({
            'source': 'rag_graph',
            'papers': rag_papers,
            'entities': entities,
            'memories': memories[:10],
        })
    
    jfr_conn.close()
    
    return {
        'manuscript': ms,
        'linked_papers': paper_links,
        'rag_results': rag_results,
    }


def search_rag(query: str, limit: int = 10) -> dict:
    """Search RAG with query across papers
    
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
            'results': []
        }
    
    from config import DB_PATH as RAG_DB_PATH
    rag_conn = sqlite3.connect(str(RAG_DB_PATH))
    rag_conn.row_factory = sqlite3.Row
    
    # Load RAG config
    import sys
    sys.path.insert(0, str(RAG_ROOT))
    import config as rag_config
    from retrieval.embed import load_embed_model, embed_query as rag_embed_query
    from retrieval.search import search as rag_search
    
    # Embed query
    tok, emb = load_embed_model()
    query_vec = rag_embed_query(tok, emb, query)
    
    # Search
    results = rag_search(query, query_vec, limit=limit)
    
    rag_conn.close()
    
    return {
        'rag_status': 'success',
        'results': results,
        'total': len(results),
        'rag_source': {
            'papers_indexed': 1470,
            'chunks_indexed': 15000,
        }
    }
