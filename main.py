import os
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# --------- Load environment (.env) ---------

# This will read variables from a .env file in the same folder
# Make sure your .env contains:  OPENAI_API_KEY=your_real_key_here
load_dotenv()


# --------- 1. PDF loading & chunking ---------

def load_pdf_text(pdf_path: str) -> str:
    """Read all text from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """
    Very simple character-based chunking.
    chunk_size: length of each chunk in characters
    overlap: how many characters to overlap between chunks
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # step back a bit to keep overlap

    return [c for c in chunks if c]


# --------- 2. Embeddings & retrieval ---------

def build_embeddings(chunks, embedder):
    """Embed each chunk into a vector."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings  # shape: (num_chunks, dim)


def cosine_sim(a, b):
    """Compute cosine similarity between two vectors (or batches)."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(question: str, chunks, chunk_embeddings, embedder, top_k: int = 3):
    """Return top_k chunks most similar to the question."""
    q_emb = embedder.encode([question], convert_to_numpy=True)
    sims = cosine_sim(q_emb, chunk_embeddings)[0]  # shape: (num_chunks,)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# --------- 3. LLM call with retrieved context ---------

def answer_with_llm(question: str, context_chunks, client: OpenAI) -> str:
    """
    Send question + retrieved context to the LLM and return the answer.
    Uses OpenAI Chat Completions API (gpt-4.1-mini by default).
    """
    context = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""
You are a question-answering assistant.

Use ONLY the information in the CONTEXT below to answer the QUESTION.
If the answer is not in the context, say you don't know.

CONTEXT:
\"\"\" 
{context}
\"\"\"

QUESTION: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # change if you want another chat model
        messages=[
            {
                "role": "system",
                "content": "You answer questions based strictly on the provided context.",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    message = response.choices[0].message
    content = message["content"] if isinstance(message, dict) else message.content
    return content.strip()


# --------- 4. Main CLI ---------

def main():
    # Check API key first so the error is clear
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "• Ensure you have a .env file in this folder containing:\n"
            "    OPENAI_API_KEY=your_real_key_here\n"
            "• Or set it in the terminal before running the script."
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Loading PDF from: {pdf_path}")
    full_text = load_pdf_text(pdf_path)
    print(f"PDF loaded. Total characters: {len(full_text)}")

    print("Chunking text...")
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (this may take a moment)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding chunks...")
    chunk_embeddings = build_embeddings(chunks, embedder)
    print("Embeddings ready.")

    print("\nRAG is ready. Ask questions about the PDF.")
    print("Type 'exit', 'quit', or Ctrl+C to stop.\n")

    # Initialise OpenAI client (reads the key from env)
    client = OpenAI(api_key=api_key)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if not question:
            continue

        # 1. Retrieve relevant chunks
        top_chunks = retrieve_relevant_chunks(
            question, chunks, chunk_embeddings, embedder, top_k=3
        )

        # 2. Ask LLM using those chunks as context
        print("Thinking...\n")
        answer = answer_with_llm(question, top_chunks, client)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()

