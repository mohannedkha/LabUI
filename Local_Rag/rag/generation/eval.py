#!/usr/bin/env python3
"""
Basic RAG evaluation using Ragas.
Requires a running Ollama server.
"""

import sys
from pathlib import Path
import json
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_BASE_URL, GEN_MODEL, DB_PATH
from retrieval.search import search
from retrieval.embed import load_embed_model, embed_query

try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
    from datasets import Dataset
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print("Please install ragas, datasets, and langchain-community: pip install ragas datasets langchain-community")
    sys.exit(1)

def generate_synthetic_qa(num_pairs=5) -> list[dict]:
    # In a real scenario, you would pull random chunks and ask the LLM to generate questions.
    # For demonstration, returning a hardcoded set based on typical scientific queries.
    return [
        {
            "question": "What is the primary mechanism stabilizing bulk nanobubbles?",
            "ground_truth": "The primary mechanism is the accumulation of surface charge forming a dense electrical double layer."
        },
        {
            "question": "How do surfactants affect enhanced oil recovery?",
            "ground_truth": "Surfactants lower the interfacial tension and alter the wettability of the rock surface."
        }
    ]

def run_evaluation():
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL} with model {GEN_MODEL}...")
    llm = ChatOllama(model=GEN_MODEL, base_url=OLLAMA_BASE_URL)
    # Ragas needs embeddings for answer relevancy. We can use Ollama.
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

    print("Loading local embeddings for retrieval...")
    tok, emb = load_embed_model()
    from retrieval.rerank import load_reranker
    rnk = load_reranker()

    qa_pairs = generate_synthetic_qa()
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for idx, pair in enumerate(qa_pairs):
        q = pair["question"]
        print(f"Querying: {q}")
        
        # 1. Retrieve contexts
        qvec = embed_query(tok, emb, q)
        chunks = search(q, qvec, rnk, top_k=5)
        contexts = [c["text"] for c in chunks]
        
        # 2. Generate answer
        prompt = f"Answer the question based only on the context.\nContext: {' '.join(contexts)}\nQuestion: {q}"
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False
        }).json()
        answer = resp.get("response", "")
        
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(pair["ground_truth"])

    dataset = Dataset.from_dict(data)
    
    print("Running Ragas evaluation...")
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\nEvaluation Results:")
    print(results)

if __name__ == "__main__":
    run_evaluation()
