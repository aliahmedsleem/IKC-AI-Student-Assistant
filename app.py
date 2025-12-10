# app.py
# -*- coding: utf-8 -*-
"""
University FAQ Chatbot (TF-IDF Version)
نظام دردشة يجيب عن الأسئلة الشائعة حول المناهج والتسجيل
باستخدام خوارزمية TF-IDF + Cosine Similarity فقط (بدون Torch).
"""

from flask import Flask, render_template, request, jsonify
import sqlite3, os, uuid
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_utils import get_alpha  # نستخدمه كإعداد بسيط لوضع النظام
from admin_routes import admin_bp

DB_PATH = "faqs.db"

app = Flask(__name__)
app.register_blueprint(admin_bp)


# ----------------- DB Helpers ----------------- #

def load_faqs():
    """تحميل الأسئلة والأجوبة من قاعدة البيانات."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, question, answer, COALESCE(category,'') FROM faqs")
    rows = c.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    questions = [(r[1] or "").strip() for r in rows]
    answers = [(r[2] or "").strip() for r in rows]
    categories = [(r[3] or "").strip() for r in rows]
    return ids, questions, answers, categories


# ----------------- NLP Helpers ----------------- #

def dynamic_threshold(text: str) -> float:
    """ضبط عتبة الثقة حسب طول السؤال."""
    n = len(text.split())
    if n <= 2:
        return 0.25
    if n <= 6:
        return 0.35
    return 0.45


def exact_or_partial_match(user_text: str, questions):
    """
    مطابقة تامة أو جزئية حسب الكلمات المشتركة.
    ترجع index للسؤال المناسب إن وجد، أو None.
    """
    t = user_text.strip().lower()
    # مطابقة تامة
    for i, q in enumerate(questions):
        if t == (q or "").strip().lower():
            return i

    # مطابقة جزئية بكلمات مهمة
    toks = [w for w in t.split() if len(w) >= 3]
    if not toks:
        return None

    best_idx = None
    best_hits = 0
    for i, q in enumerate(questions):
        ql = (q or "").lower()
        hits = sum(1 for tok in toks if tok in ql)
        if hits > best_hits and hits > 0:
            best_hits = hits
            best_idx = i

    return best_idx


def build_tfidf(questions):
    """بناء نموذج TF-IDF لمجموعة الأسئلة."""
    if not questions:
        return None, None
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\w+",
        ngram_range=(1, 2),
        max_features=10000,
    )
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix


def tfidf_search(user_text, questions):
    """
    استخدام TF-IDF + cosine similarity لاسترجاع أقرب سؤال.
    ترجع (indices_sorted, scores_sorted)
    """
    if not questions:
        return [], np.array([])

    vectorizer, tfidf_matrix = build_tfidf(questions)
    query_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # shape: (n_questions,)

    idxs = np.argsort(-scores)  # أكبر تشابه أولاً
    return idxs, scores


# ----------------- Routes ----------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    from_suggestion = bool(data.get("from_suggestion") or False)

    if not user_message:
        return jsonify({"reply": "الرجاء إدخال سؤال.", "suggestions": []})

    ids, questions, answers, _ = load_faqs()
    if not questions:
        return jsonify({"reply": "لا توجد بيانات حالياً في النظام.", "suggestions": []})

    # وضع النظام (يجي من config_utils & admin)
    alpha = get_alpha()
    # نستخدمه لتحديد نمط أكثر تحفظاً أو أكثر تساهلاً في العتبة
    base_thr = dynamic_threshold(user_message)
    if alpha < 0.7:
        # High-Recall mode → نخفّض العتبة قليلاً
        thr = max(0.15, base_thr - 0.1)
    else:
        thr = base_thr

    # بحث TF-IDF
    idxs, scores = tfidf_search(user_message, questions)

    reply = ""
    suggestions = []
    matched_id = None
    confidence = 0.0

    if len(idxs) > 0:
        best_idx = int(idxs[0])
        confidence = float(scores[best_idx])

        if from_suggestion:
            # إذا السؤال جاء من زر اقتراح، نثق بالنتيجة مباشرة
            reply = answers[best_idx]
            matched_id = ids[best_idx]
        else:
            # أولاً: نحاول مطابقة تامة / جزئية سهلة
            fb = exact_or_partial_match(user_message, questions)
            if fb is not None:
                reply = answers[fb]
                matched_id = ids[fb]
                confidence = max(confidence, 0.95)
            else:
                if confidence >= thr:
                    reply = answers[best_idx]
                    matched_id = ids[best_idx]
                else:
                    # اقتراحات إضافية لأعلى 3 أسئلة
                    alt = []
                    for i in idxs[1:4]:
                        i = int(i)
                        if 0 <= i < len(questions):
                            alt.append(questions[i])
                    if alt:
                        reply = "لربما تقصد احد هذه الاسئلة ! اختر أحد الاقتراحات أو أعد صياغة السؤال."
                        suggestions = alt
                    else:
                        reply = "سؤالك جدا مهم لذلك سيتم اخذ سؤالك بمنتهى الجدية وايصاله الى القسم المختص نرجوا منك المحاولة مرة ثانية فيما بعد لتوفير الاجابة على سؤال"
    else:
        reply = "سؤالك جدا مهم لذلك سيتم اخذ سؤالك بمنتهى الجدية وايصاله الى القسم المختص نرجوا منك المحاولة مرة ثانية فيما بعد لتوفير الاجابة على سؤال"

    # تسجيل المحادثة في قاعدة البيانات
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
                float(confidence),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ خطأ أثناء حفظ المحادثة:", e)

    return jsonify({"reply": reply, "suggestions": suggestions})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
