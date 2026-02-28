# rag_engine.py
# -*- coding: utf-8 -*-

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = "chroma_faqs"
COLLECTION = "faqs"

# كلما قلّت MAX_DISTANCE = يصير النظام أكثر حذر (أدق وأقل عشوائية)
# جرب 0.35 إلى 0.50
MAX_DISTANCE = 0.50

EMBED_MODEL = "bge-m3"  # لازم نفس rag_index.py


def build_rag_chain():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://127.0.0.1:11434",
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    def answer(question: str):
        # نجيب أفضل 3 نتائج (ونقرر من أول نتيجة)
        results = vectordb.similarity_search_with_score(question, k=3)

        if not results:
            return ("عذرًا، لا أملك معلومات كافية للإجابة. راجع شعبة التسجيل.", None, None)

        best_doc, distance = results[0]  # distance أقل = أفضل

        faq_id = best_doc.metadata.get("faq_id")
        faq_answer = (best_doc.metadata.get("answer") or "").strip()

        print("🔎 Question:", question)
        print("📏 Distance:", distance)
        print("🧾 Matched FAQ ID:", faq_id)
        print("-" * 50)

        # إذا التشابه ضعيف أو لا يوجد جواب محفوظ
        if distance is None or distance > MAX_DISTANCE or not faq_answer:
            return ("سؤالك مهم ✅ تم تسجيله للمراجعة. جرّب صياغة أخرى أو راجع شعبة التسجيل.", None, float(distance) if distance is not None else None)

        # نرجع الجواب الأصلي 100% (بدون توليد)
        return (faq_answer, faq_id, float(distance))

    return answer