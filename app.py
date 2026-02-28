# app.py
# -*- coding: utf-8 -*-
"""
University FAQ Chatbot (RAG Version - Stable)
نظام دردشة يجيب عن الأسئلة الشائعة حول المناهج والتسجيل
باستخدام RAG (Ollama + Chroma).

- Retrieval: Chroma Vector Store (persisted in chroma_faqs/)
- Embeddings: bge-m3 أو nomic-embed-text (حسب rag_engine/rag_index)
- Logs: conversations table in faqs.db
"""

from flask import Flask, render_template, request, jsonify
import sqlite3, os, uuid
from datetime import datetime

from admin_routes import admin_bp
from rag_engine import build_rag_chain

DB_PATH = "faqs.db"

app = Flask(__name__)
app.register_blueprint(admin_bp)

# نبني دالة RAG مرة واحدة عند تشغيل السيرفر
print("🧠 Initializing RAG engine (Ollama + Chroma)...")
rag_answer = build_rag_chain()  # callable: rag_answer(question)
print("✅ RAG engine is ready.")


# ----------------- Routes ----------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()

    # للتوافق مع واجهتك
    _from_suggestion = bool(data.get("from_suggestion") or False)

    if not user_message:
        return jsonify({"reply": "الرجاء إدخال سؤال.", "suggestions": []})

    reply = ""
    suggestions = []  # حالياً بدون اقتراحات لتجنب اللوب
    matched_id = None
    confidence = None  # سنخزن distance هنا (إذا متوفر) أو None

    # ----------------- RAG Answer (Robust) ----------------- #
    try:
        res = rag_answer(user_message)

        # ✅ مرونة: rag_answer قد يرجع tuple/list أو string
        if isinstance(res, (list, tuple)):
            reply = (res[0] or "").strip() if len(res) > 0 else ""
            matched_id = res[1] if len(res) > 1 else None
            confidence = res[2] if len(res) > 2 else None
            # إذا كانت هناك قيمة رابعة (docs/context) نتجاهلها
        else:
            reply = str(res).strip()
            matched_id = None
            confidence = None

        if not reply:
            reply = "عذرًا، لم أتمكن من استخراج إجابة الآن. حاول مرة أخرى."

    except Exception as e:
        print("⚠️ RAG error:", e)
        reply = "حدثت مشكلة داخل محرك الاسترجاع. أعد تشغيل السيرفر ثم جرّب."
        matched_id = None
        confidence = None

    # ----------------- Save Conversation ----------------- #
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        session_id = str(uuid.uuid4())
        c.execute(
            """
            INSERT INTO conversations (session_id, timestamp, user_message, bot_reply, matched_faq_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now().isoformat(),
                user_message,
                reply,
                matched_id,
                confidence if confidence is None else float(confidence),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ DB log error:", e)

    return jsonify({"reply": reply, "suggestions": suggestions})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # مهم: نطفي reloader حتى لا يعيد تهيئة vectordb مرتين
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)