#!/usr/bin/env python3
"""
SecondSelf — Phase 2 Semantic Search Utility

Allows you to search your wiki notes semantically using the embeddings 
generated in Phase 2.

Usage:
    python search.py "your search query" [--top-k 5]
"""

import argparse
from lib import embeddings, storage


def search(query: str, top_k: int = 5):
    print(f"Searching for: '{query}'...")
    
    # 1. Embed the query
    query_vec = embeddings.embed_text(query)
    
    # 2. Load all embeddings
    all_embeddings = embeddings.load_embeddings()
    if not all_embeddings:
        print("No embeddings found. Please run `python pipeline.py process` first.")
        return
        
    # 3. Read all wiki notes to get their details
    notes = {note["id"]: note for note in storage.read_wiki_notes()}
    
    # 4. Compute similarities
    results = []
    for note_id, note_vec in all_embeddings.items():
        if note_id in notes:
            sim = embeddings.cosine_similarity(query_vec, note_vec)
            results.append((note_id, sim))
            
    # 5. Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Display results
    print("\nTop Results:")
    print("=" * 60)
    for i, (note_id, score) in enumerate(results[:top_k], 1):
        note = notes[note_id]
        print(f"{i}. [{note['para']}] {note['summary']} (ID: {note_id}, Similarity: {score:.3f})")
        print(f"   Path: {note['path']}")
        print(f"   Tags: {', '.join(note.get('tags', []))}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Semantic search across wiki notes.")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    search(args.query, args.top_k)


if __name__ == "__main__":
    main()
