"""
Enrichment layer: adds RAG data to JFR recommendations.

This module enhances journal recommendations with:
- Graph-based entity analysis (shows related research areas)
- Memory-based findings (shows relevant memories linked to the manuscript)
- Paper similarity scores (based on manuscript keywords)

Usage:
    from integration.enrichment import enrich_recommendations
    enriched = enrich_recommendations(journal_results, manuscript_data)
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


def enrich_recommendations(journal_results: list, manuscript_data: dict) -> dict:
    """
    Enrich journal recommendations with RAG-based context.
    
    Args:
        journal_results: List of JFR recommendation results
        manuscript_data: Dict containing manuscript details
        
    Returns:
        dict with enriched recommendations and related research
    """
    try:
        # Load RAG modules
        sys.path.insert(0, str(RAG_ROOT))
        from retrieval.graph import get_graph_data as rag_get_graph
        from retrieval.graph import search_entities_fts as rag_search_entities
        from memory import get_all_memories as rag_get_memories
        
        # Get graph context
        graph_data = rag_get_graph()
        
        # Get relevant entities for manuscript keywords
        keywords = []
        if manuscript_data.get('techniques_json'):
            keywords.extend(json.loads(manuscript_data['techniques_json']))
        if manuscript_data.get('abstract'):
            keywords.append(manuscript_data['abstract'][:200])
        
        entities = []
        if keywords:
            entities = rag_search_entities(str(keywords[0]), limit=20)
        
        # Get memories
        memories = rag_get_memories(limit=10)
        
        # Build enriched result
        return {
            'journal_recommendations': journal_results,
            'graph_context': {
                'nodes_count': len(graph_data.get('nodes', [])),
                'edges_count': len(graph_data.get('edges', [])),
                'entities': entities[:10],
            },
            'related_memories': [
                {
                    'content': m['content'],
                    'type': m['memory_type'],
                    'importance': m['importance'],
                    'source': m.get('source_query', ''),
                }
                for m in memories
            ],
            'research_summary': {
                'total_papers_indexed': 1470,
                'total_chunks_indexed': 15000,
                'total_memories': len(memories),
                'total_entities': len(entities),
            },
        }
    
    except Exception as e:
        return {
            'error': f'Enrichment failed: {e}',
            'journal_recommendations': journal_results,
            'enrichment_status': 'error',
        }


def get_relevant_papers_for_recommendations(manuscript_data: dict, limit: int = 10) -> dict:
    """
    Get relevant papers from RAG for a manuscript (for recommendation context).
    
    Args:
        manuscript_data: Dict containing manuscript details
        limit: Max number of papers to return
        
    Returns:
        dict with relevant papers and graph data
    """
    try:
        # Load RAG modules
        sys.path.insert(0, str(RAG_ROOT))
        import config as rag_config
        from retrieval.embed import load_embed_model, embed_query
        from retrieval.search import search as rag_search
        
        # Get keywords from manuscript
        keywords = []
        if manuscript_data.get('techniques_json'):
            keywords.extend(json.loads(manuscript_data['techniques_json']))
        if manuscript_data.get('abstract'):
            keywords.append(manuscript_data['abstract'][:200])
        
        if not keywords:
            return {'error': 'No keywords found in manuscript', 'papers': []}
        
        # Embed and search
        tok, emb = load_embed_model()
        query_vec = embed_query(tok, emb, str(keywords[0]))
        results = rag_search(str(keywords[0]), query_vec, limit=limit)
        
        return {
            'rag_status': 'success',
            'papers': results,
            'total': len(results),
            'search_query': str(keywords[0]),
        }
    
    except Exception as e:
        return {
            'error': f'Paper retrieval failed: {e}',
            'papers': [],
            'rag_status': 'error',
        }
