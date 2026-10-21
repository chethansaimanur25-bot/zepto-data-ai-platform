import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "support_assistant/docs"
DB_DIR = "support_assistant/chroma_db"
COLLECTION = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Load the local embedding model once (no API, no account)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


def load_chunks():
    """Ingestion + chunking: one document per chunk, chunk id = document id."""
    chunks = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
                chunks.append({"id": fname[:-4], "text": f.read().strip()})
    return chunks


def ingest(force=False):
    """Embedding: encode every chunk and store it in ChromaDB."""
    col = get_collection()
    if col.count() > 0 and not force:
        return col
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, normalize_embeddings=True).tolist()
    col.upsert(ids=[c["id"] for c in chunks], documents=texts,
               embeddings=embeddings)
    return col


def retrieve(query, k=3):
    """Retrieval: embed the query and return the top-k chunks by cosine similarity."""
    col = ingest()
    q = get_model().encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    return [
        {"id": res["ids"][0][i],
         "text": res["documents"][0][i],
         "similarity": round(1 - res["distances"][0][i], 4)}
        for i in range(len(res["ids"][0]))
    ]



if __name__ == "__main__":
    col = ingest(force=True)
    print("collection:", COLLECTION, "| chunks stored:", col.count())


    DOCS_DIR = "support_assistant/docs"
    DB_DIR = "support_assistant/chroma_db"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    DB_DIR = os.path.join(BASE_DIR, "chroma_db")