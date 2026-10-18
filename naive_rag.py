
from openai import OpenAI
import chromadb

oai = OpenAI()

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Batch embed texts with OpenAI."""
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    resp = oai.embeddings.create(input=cleaned, model=model)
    return [d.embedding for d in resp.data]

def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]


# --- Chunk all documents ---

all_chunks = []
chunk_meta = []

for doc in DOCUMENTS:
    chunks = chunk_recursive(doc["content"], max_size=100)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_meta.append({"title": doc["title"], "source": doc["source"]})

print(f"Total chunks: {len(all_chunks)}")
for doc in DOCUMENTS:
    n = sum(1 for m in chunk_meta if m["title"] == doc["title"])
    print(f"  {doc['title']}: {n} chunks")


# --- Embed and store in ChromaDB ---

chroma = chromadb.Client()

try: chroma.delete_collection("naive_rag")
except: pass

collection = chroma.create_collection("naive_rag", metadata={"hnsw:space": "cosine"})

embs = get_embeddings(all_chunks)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    embeddings=embs,
    documents=all_chunks,
    metadatas=chunk_meta
)

print(f"✅ Stored {len(all_chunks)} chunks in ChromaDB")


# --- The naive RAG query function ---

def naive_rag(question: str, k: int = 5, verbose: bool = True) -> str:
    """Simplest RAG: semantic search → stuff prompt → generate."""

    results = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if verbose:
        print(f"\n🔍 Query: '{question}'")
        print(f"\nRetrieved {k} chunks:")
        for i, (d, m, dist) in enumerate(zip(docs, metas, dists)):
            print(f"  [{i+1}] dist={dist:.3f} | {m['title']}")
            print(f"      {d[:80]}...")

    context = "\n".join(
        f"[Source: {m['title']}]\n{d}" for d, m in zip(docs, metas)
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