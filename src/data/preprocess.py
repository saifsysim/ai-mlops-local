"""
Phase 1 — Data Preprocessing
Cleans and normalises raw JSON articles & Q&A pairs into processed/.
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip known artefacts."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def preprocess_articles() -> list[dict]:
    raw = json.loads((RAW_DIR / "articles.json").read_text())
    cleaned = []
    for article in raw:
        cleaned.append(
            {
                "id": article["id"],
                "title": clean_text(article["title"]),
                "content": clean_text(article["content"]),
                "source": article.get("source", "unknown"),
                "tags": article.get("tags", []),
                # Word count — useful for chunking decisions
                "word_count": len(clean_text(article["content"]).split()),
            }
        )
    print(f"✅ Preprocessed {len(cleaned)} articles")
    return cleaned


def preprocess_qa_pairs() -> list[dict]:
    raw = json.loads((RAW_DIR / "qa_pairs.json").read_text())
    cleaned = []
    for qa in raw:
        cleaned.append(
            {
                "id": qa["id"],
                "question": clean_text(qa["question"]),
                "answer": clean_text(qa["answer"]),
                "context_ids": qa.get("context_ids", []),
            }
        )
    print(f"✅ Preprocessed {len(cleaned)} Q&A pairs")
    return cleaned


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    articles = preprocess_articles()
    (PROCESSED_DIR / "articles_clean.json").write_text(
        json.dumps(articles, indent=2)
    )

    qa_pairs = preprocess_qa_pairs()
    (PROCESSED_DIR / "qa_pairs_clean.json").write_text(
        json.dumps(qa_pairs, indent=2)
    )

    print("✅ Preprocessing complete → data/processed/")


if __name__ == "__main__":
    main()
