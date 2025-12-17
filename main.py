import os
import sys
import re
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 900      # characters per chunk (simple + robust)
CHUNK_OVERLAP = 150   # overlap to preserve context
TOP_K = 5             # how many chunks to retrieve


# ----------------------------
# Utilities
# ----------------------------
def read_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF (best-effort)."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    text = "\n".join(pages_text)
    # Normalize whitespace a bit
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunking by characters."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Embed a list of texts and return a (N, D) numpy array."""
    # OpenAI embeddings endpoint accepts batched inputs
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between query_vec (D,) and doc_vecs (N, D)."""
    # Normalize
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return (d @ q).astype(np.float32)  # (N,)


def build_context(chunks: List[str], scores: np.ndarray, top_k: int = TOP_K) -> Tuple[str, List[int]]:
    """Select top-k chunks and build a context string."""
    top_idx = np.argsort(scores)[::-1][:top_k]
    selected = []
    for rank, i in enumerate(top_idx, start=1):
        selected.append(f"[Chunk {i} | score={scores[i]:.3f}]\n{chunks[i]}")
    return "\n\n---\n\n".join(selected), top_idx.tolist()


# ----------------------------
# RAG QA
# ----------------------------
def answer_question(client: OpenAI, question: str, chunks: List[str], chunk_vecs: np.ndarray) -> str:
    # Embed query
    q_vec = embed_texts(client, [question])[0]  # (D,)
    sims = cosine_sim_matrix(q_vec, chunk_vecs)  # (N,)

    context, top_ids = build_context(chunks, sims, TOP_K)

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say: 'I don't know based on the provided PDF.' "
        "Be concise, and when helpful, quote short phrases from the context."
    )

    user = f"""CONTEXT (retrieved from PDF):
{context}

QUESTION:
{question}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------
# CLI
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python rag_pdf_cli.py "C:\\path\\to\\file.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Put it in a .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading PDF from: {pdf_path}")
    text = read_pdf_text(pdf_path)
    if not text:
        print("Error: No text could be extracted from this PDF.")
        sys.exit(1)

    print(f"PDF loaded. Characters: {len(text)}")
    print("Chunking text...")
    chunks = chunk_text(text)
    if not chunks:
        print("Error: Chunking produced no chunks.")
        sys.exit(1)

    print(f"Created {len(chunks)} chunks.")
    print("Embedding chunks (this may take a moment)...")
    chunk_vecs = embed_texts(client, chunks)  # (N, D)

    print("\nRAG is ready. Ask questions about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            ans = answer_question(client, q, chunks, chunk_vecs)
            print("\nAnswer:\n" + ans + "\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()


