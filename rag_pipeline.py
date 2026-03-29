import os
from dotenv import load_dotenv
load_dotenv()
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from pypdf import PdfReader

@dataclass
class Chunk:
    id: str
    text: str

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()

def load_source_document(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    if path.lower().endswith(".txt"):
        return load_text_file(path)

    if path.lower().endswith(".pdf"):
        return load_pdf_file(path)

    raise ValueError("Unsupported file format. Use .txt or .pdf")

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_by_sections(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    current_title = "Preamble"
    buffer: List[str] = []

    section_header = re.compile(
        r"^\s*((Section\s+\d+[:.]?)|(\d+(\.\d+)*[.)]?))\s+(.+?)\s*$",
        re.IGNORECASE,
    )

    def flush() -> None:
        nonlocal buffer, current_title
        content = "\n".join(buffer).strip()
        if content:
            chunks.append(Chunk(id=current_title, text=content))
        buffer = []

    for line in lines:
        match = section_header.match(line)
        if match:
            flush()
            current_title = line.strip()
            buffer.append(line.strip())
        else:
            buffer.append(line.strip())

    flush()
    return [chunk for chunk in chunks if chunk.text.strip()]

class ContractRAG:
    def __init__(self, source_path: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)

        self.source_text = normalize_text(load_source_document(source_path))
        self.chunks = chunk_by_sections(self.source_text)

        if not self.chunks:
            raise ValueError("No valid chunks were created from the source document.")

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.X = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def retrieve(self, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.X).flatten()
        idxs = sims.argsort()[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            results.append(
                {
                    "idx": int(i),
                    "score": float(sims[i]),
                    "id": self.chunks[i].id,
                    "text": self.chunks[i].text,
                }
            )
        return results

    def build_prompt(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[{r['id']}] (score={r['score']:.3f})\n{r['text']}"
            for r in retrieved
        )

        return f"""
You are a Contract Auditor.

RULES:
- Answer ONLY using the context below.
- If the answer is not clearly stated in the context, reply exactly:
I cannot find this in the provided contract text.
- Do not use outside knowledge.
- Keep the answer concise.
- Cite the section IDs you used.

CONTEXT:
{context}

QUESTION:
{question}

FORMAT:
Answer:
Evidence (Section IDs):
""".strip()

    def answer(self, question: str, top_k: int = 4, min_score: float = 0.12) -> str:
        retrieved = self.retrieve(question, top_k=top_k)

        if not retrieved or retrieved[0]["score"] < min_score:
            return "I cannot find this in the provided contract text."

        prompt = self.build_prompt(question, retrieved)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 300,
                "top_p": 0.9,
            },
        )

        text = (response.text or "").strip()

        if not text:
            return "I cannot find this in the provided contract text."

        if "I cannot find this in the provided contract text." not in text:
            if "Evidence" not in text:
                used = sorted({r["id"] for r in retrieved})
                text += f"\n\nEvidence (Section IDs): {used}"

        return text