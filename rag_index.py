# rag_index.py
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import sqlite3

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_PATH = "faqs.db"
CHROMA_DIR = "chroma_faqs"
COLLECTION = "faqs"
CONFIG_FILE = "config.json"

# ===== Embedding settings =====
EMBEDDING_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_LOCAL_DIR = os.getenv(
    "HF_EMBEDDING_LOCAL_DIR",
    os.path.join("models", "BAAI__bge-m3")
)
HF_OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

DEFAULT_CONFIG = {
    "embedding_model": "BAAI/bge-m3",
    "top_k": 10,
    "alpha": 0.55,
    "min_keyword_coverage": 0.15,
    "min_combined_score": 0.22,
    "max_distance": 2.20
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def table_exists(conn, name: str) -> bool:
    c = conn.cursor()
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return c.fetchone() is not None


def build_search_text(question: str, answer: str, category: str, paraphrase: str) -> str:
    q = (question or "").strip()
    a = (answer or "").strip()
    c = (category or "").strip()
    p = (paraphrase or "").strip()

    parts = []

    if q:
        parts.append(f"السؤال: {q}")
        parts.append(f"صياغة منقحة: {normalize_text(q)}")

    if p:
        parts.append(f"صياغات بديلة: {p}")
        for item in [x.strip() for x in p.split('||') if x.strip()]:
            parts.append(f"paraphrase: {item}")
            parts.append(f"صياغة منقحة بديلة: {normalize_text(item)}")

    if c:
        parts.append(f"التصنيف: {c}")

    if a:
        parts.append(f"الجواب: {a}")

    return "\n".join(parts).strip()


def _get_embedding_source(config_model_name: str) -> str:
    """
    الأولوية:
    1) المسار المحلي إذا موجود
    2) اسم الموديل من config / env للتحميل الأونلاين
    """
    if os.path.isdir(EMBEDDING_LOCAL_DIR):
        print(f"✅ Using local embedding model: {EMBEDDING_LOCAL_DIR}")
        return EMBEDDING_LOCAL_DIR

    model_name = config_model_name or EMBEDDING_MODEL_NAME
    print(f"🌐 Using remote embedding model: {model_name}")
    return model_name


def _build_embeddings(config_model_name: str):
    model_source = _get_embedding_source(config_model_name)

    if HF_OFFLINE_MODE and not os.path.isdir(EMBEDDING_LOCAL_DIR):
        raise RuntimeError(
            f"❌ HF_HUB_OFFLINE=1 but local embedding model not found at: {EMBEDDING_LOCAL_DIR}"
        )

    print("🧠 Loading embedding model:", model_source)
    return HuggingFaceEmbeddings(
        model_name=model_source,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(reset: bool = True):
    cfg = load_config()
    embedding_model = cfg.get("embedding_model", EMBEDDING_MODEL_NAME)

    if not os.path.exists(DB_PATH):
        raise RuntimeError("❌ ملف faqs.db غير موجود.")

    conn = sqlite3.connect(DB_PATH)

    if not table_exists(conn, "faqs"):
        conn.close()
        raise RuntimeError("❌ جدول faqs غير موجود داخل faqs.db.")

    c = conn.cursor()
    c.execute("""
        SELECT
            id,
            question,
            answer,
            COALESCE(category,''),
            COALESCE(paraphrase,'')
        FROM faqs
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("⚠️ جدول faqs فارغ.")

    docs = []
    ids = []

    for faq_id, q, a, cat, paraphrase in rows:
        q = (q or "").strip()
        a = (a or "").strip()
        cat = (cat or "").strip()
        paraphrase = (paraphrase or "").strip()

        if not q:
            continue

        page_content = build_search_text(q, a, cat, paraphrase)

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "faq_id": int(faq_id),
                    "question": q,
                    "answer": a,
                    "category": cat,
                    "paraphrase": paraphrase,
                }
            )
        )
        ids.append(f"faq_{int(faq_id)}")

    if not docs:
        raise RuntimeError("⚠️ لا توجد أسئلة صالحة للفهرسة.")

    embeddings = _build_embeddings(embedding_model)

    if reset and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )

    print(f"✅ Index rebuilt successfully with {len(docs)} FAQ records.")
    print(f"📁 Folder: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index(reset=True)