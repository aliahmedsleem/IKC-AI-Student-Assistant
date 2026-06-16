# -*- coding: utf-8 -*-
import re

# =========================
# 🔹 إعدادات عامة
# =========================
MIN_SCORE = 0.45
FALLBACK_OVERLAP = 2

# =========================
# 🔹 Synonyms (مهم جداً)
# =========================
SYNONYMS = {
    "معدل": ["معدل", "قبول", "نسبة"],
    "اجور": ["اجور", "رسوم", "قسط", "تكلفة"],
    "مسائي": ["مسائي", "مساء", "دوام مسائي"],
    "صباحي": ["صباحي", "صباح"],
    "تقديم": ["تقديم", "تسجيل", "اونلاين", "الكتروني"],
    "نقل": ["نقل", "باص", "خطوط"],
    "قانون": ["قانون"],
    "هندسة": ["هندسه", "هندسة"],
    "حاسوب": ["حاسوب", "حاسبات"],
}

# =========================
# 🔹 تنظيف النص
# =========================
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text

# =========================
# 🔹 تحويل النص إلى كلمات
# =========================
def tokenize(text):
    text = normalize(text)
    return set(text.split())

# =========================
# 🔹 توسيع الكلمات بالمرادفات
# =========================
def expand_terms(words):
    expanded = set(words)

    for word in words:
        for key, values in SYNONYMS.items():
            if word in values:
                expanded.update(values)

    return expanded

# =========================
# 🔹 حساب التشابه
# =========================
def compute_score(q_terms, db_terms):
    overlap = len(q_terms & db_terms)
    coverage = overlap / (len(q_terms) + 1e-5)

    score = (overlap * 0.6) + (coverage * 0.4)

    return score, overlap, coverage

# =========================
# 🔹 الماتشر الرئيسي
# =========================
def match_question(user_question, faq_list):
    q_tokens = tokenize(user_question)
    q_terms = expand_terms(q_tokens)

    best_score = 0
    best_match = None
    best_overlap = 0

    for faq in faq_list:
        db_tokens = tokenize(faq["question"])
        db_terms = expand_terms(db_tokens)

        score, overlap, coverage = compute_score(q_terms, db_terms)

        if score > best_score:
            best_score = score
            best_match = faq
            best_overlap = overlap

    # =========================
    # 🔥 القرار النهائي
    # =========================

    # 1. تطابق قوي
    if best_score >= MIN_SCORE:
        return {
            "status": "matched",
            "data": best_match
        }

    # 2. fallback ذكي (هذا سر القوة)
    if best_overlap >= FALLBACK_OVERLAP:
        return {
            "status": "fallback",
            "data": best_match
        }

    # 3. اقتراحات
    suggestions = []
    for faq in faq_list[:3]:
        suggestions.append(faq["question"])

    return {
        "status": "no_match",
        "suggestions": suggestions
    }