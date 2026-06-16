"""
Integration layer between RAG (literature) and JFR (publication) systems.

This module bridges two previously isolated systems:
- RAG: literature search, paper embeddings, knowledge graph, memories
- JFR: manuscript tracking, journal recommendations, submission states

Usage:
    from integration import unified_search
    results = unified_search("nanobubble stability")
    
"""

__all__ = [
    'rag_integration',
    'search',
    'enrichment',
    'linking',
    'lab_view',
]
