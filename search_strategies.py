
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())

# Build BM25 index over the same chunks
bm25 = BM25Okapi([tokenize(c) for c in all_chunks])

print(f"✅ Built BM25 index over {len(all_chunks)} chunks")


def reciprocal_rank_fusion(
    semantic: List[Tuple[int, float]],
    keyword: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Merge two ranked lists with RRF.
    Simple, effective, no hyperparameters to tune.
    """
    scores = {}
    for rank, (idx, _) in enumerate(semantic):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(keyword):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(question: str, k: int = 5) -> List[Dict]:
    """Combine semantic + BM25 with RRF."""

    # Semantic search (broad, top 20)
    sem = collection.query(query_embeddings=[get_embedding(question)], n_results=20)
    sem_ranked = [
        (int(id.split("_")[1]), dist)
        for id, dist in zip(sem["ids"][0], sem["distances"][0])
    ]

    # BM25 keyword search (top 20)
    scores = bm25.get_scores(tokenize(question))
    bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:20]

    # Fuse
    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked)

    return [
        {"chunk": all_chunks[idx], "meta": chunk_meta[idx], "score": sc}
        for idx, sc in fused[:k]
    ]


def hybrid_rag(question: str, k: int = 5, verbose: bool = True) -> str:
    """RAG with hybrid search."""
    results = hybrid_search(question, k)

    if verbose:
        print(f"\n🔍 Query: '{question}'")
        print(f"\nRetrieved {len(results)} chunks (hybrid):")
        for i, r in enumerate(results):
            print(f"  [{i+1}] RRF={r['score']:.4f} | {r['meta']['title']}")
            print(f"      {r['chunk'][:80]}...")

    context = "\n".join(
        f"[Source: {r['meta']['title']}]\n{r['chunk']}" for r in results
    )

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": f"""Answer based ONLY on the context below.
If the context doesn't have the answer, say \"I don't have enough information.\"
Cite your sources.

Context:
{context}

Question: {question}

Answer:"""}]
    )

    answer = resp.choices[0].message.content
    if verbose:
        print(f"\n💬 Answer:\n{answer}")
    return answer