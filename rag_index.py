# rag_index.py
# -*- coding: utf-8 -*-

import os
import shutil
import sqlite3

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

DB_PATH = "faqs.db"
CHROMA_DIR = "chroma_faqs"
COLLECTION = "faqs"

# ✅ الأفضل للعربي + إنكليزي: bge-m3 (متعدد اللغات)
# قبلها: ollama pull bge-m3
EMBED_MODEL = "bge-m3"  # أو "nomic-embed-text" إذا تريد تبقى عليه


def _table_exists(conn, name: str) -> bool:
    c = conn.cursor()
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return c.fetchone() is not None


def build_index(reset: bool = True):
    # 1) اقرأ البيانات من SQLite
    conn = sqlite3.connect(DB_PATH)
    if not _table_exists(conn, "faqs"):
        conn.close()
        raise RuntimeError("❌ جدول faqs غير موجود داخل faqs.db. تأكد من إنشاء قاعدة البيانات أولاً.")

    c = conn.cursor()
    c.execute("SELECT id, question, answer, COALESCE(category,'') FROM faqs")
    rows = c.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("⚠️ جدول faqs فارغ. أضف أسئلة/أجوبة أولاً ثم أعد بناء الـ Index.")

    # 2) جهّز Documents
    docs = []
    doc_ids = []

    for faq_id, q, a, cat in rows:
        q = (q or "").strip()
        a = (a or "").strip()
        cat = (cat or "").strip()

        if not q:
            continue

        docs.append(
            Document(
                # ✅ نخزن السؤال فقط لرفع جودة الاسترجاع
                page_content=q,
                metadata={
                    "faq_id": int(faq_id),
                    "answer": a,
                    "category": cat
                }
            )
        )
        doc_ids.append(f"faq_{int(faq_id)}")

    if not docs:
        raise RuntimeError("⚠️ لا توجد أسئلة صالحة للفهرسة (كلها فارغة).")

    # 3) Embeddings
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434"
    )

    # 4) Reset / clean old index (مهم جداً لمنع اختلاط القديم بالجديد)
    if reset:
        # أسهل وأضمن: امسح مجلد chroma_faqs بالكامل
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # 5) Build new index
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=doc_ids,  # ✅ IDs ثابتة
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION
    )

    print(f"✅ Index rebuilt successfully with {len(docs)} FAQ questions.")
    print(f"📌 Embedding model: {EMBED_MODEL}")
    print(f"📁 Chroma dir: {CHROMA_DIR}")
    print(f"📦 Collection: {COLLECTION}")


if __name__ == "__main__":
    # reset=True مهم أول مرة أو بعد تغيير EMBED_MODEL
    build_index(reset=True)