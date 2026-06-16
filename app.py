# app.py
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import uuid
from datetime import datetime

from admin_routes import admin_bp
from rag_engine import build_rag_chain

DB_PATH = "faqs.db"

app = Flask(__name__)
app.register_blueprint(admin_bp)

print("🧠 Initializing RAG engine...")
rag_answer = build_rag_chain()
print("✅ RAG engine is ready.")


# =========================
# 🔹 الصفحة الرئيسية
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# 🔹 API CHAT (النسخة النهائية)
# =========================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({
            "reply": "الرجاء إدخال سؤال.",
            "suggestions": []
        })

    try:
        result = rag_answer(user_message)

        reply = (result.get("reply") or "").strip()
        suggestions = result.get("suggestions") or []
        matched_id = result.get("matched_id")
        confidence = result.get("confidence")

        # =========================
        # 🔥 فلتر الأمان النهائي
        # =========================

        # 1. إذا ماكو جواب
        if not reply:
            reply = "سؤالك لم يطابق أيًا من بياناتنا، حاول إعادة الصياغة."

        # 2. إذا الثقة ضعيفة جداً
        if confidence is not None:
            try:
                confidence = float(confidence)

                if confidence < 0.25:
                    reply = "سؤالك لم يطابق أيًا من بياناتنا، يستحسن أن تزيد من تفاصيل سؤالك ليتسنى لنا خدمتك بصورة أفضل."
                    suggestions = []

            except:
                confidence = None

    except Exception as e:
        print("⚠️ chat error:", e)
        reply = "حدثت مشكلة داخل النظام أثناء معالجة السؤال."
        suggestions = []
        matched_id = None
        confidence = None

    # =========================
    # 🔹 تسجيل المحادثة
    # =========================
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        )
        has_conversations = c.fetchone() is not None

        if has_conversations:
            session_id = str(uuid.uuid4())

            c.execute(
                """
                INSERT INTO conversations
                (session_id, timestamp, user_message, bot_reply, matched_faq_id, confidence)
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
        print("⚠️ log error:", e)

    return jsonify({
        "reply": reply,
        "suggestions": suggestions
    })


# =========================
# 🔹 تشغيل السيرفر
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)