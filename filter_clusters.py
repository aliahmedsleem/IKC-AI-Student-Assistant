# filter_clusters.py
# -*- coding: utf-8 -*-

import json
import re
from difflib import SequenceMatcher

INPUT_FILE = "generated_clusters/question_clusters_review.json"
OUTPUT_DIR = "filtered_clusters"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# أدوات مساعدة
# ===============================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extract_keywords(text):
    words = normalize(text).split()
    return set(words)


# ===============================
# منطق التقييم
# ===============================

def classify_variant(question, variant):
    q_norm = normalize(question)
    v_norm = normalize(variant)

    if not v_norm or len(v_norm) < 3:
        return "reject"

    sim = similarity(q_norm, v_norm)

    q_words = extract_keywords(q_norm)
    v_words = extract_keywords(v_norm)

    overlap = len(q_words & v_words)

    # ===========================
    # قواعد القرار
    # ===========================

    # 🔴 مرفوض
    if overlap == 0:
        return "reject"

    if sim < 0.25:
        return "reject"

    # 🟡 مراجعة
    if overlap == 1:
        return "review"

    if sim < 0.5:
        return "review"

    # 🟢 مقبول
    return "accept"


# ===============================
# تشغيل الفلترة
# ===============================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

accepted = []
review = []
rejected = []

for item in data:
    question = item["canonical_question"]
    answer = item["answer"]
    variants = item.get("generated_variants", [])

    acc = []
    rev = []
    rej = []

    for v in variants:
        result = classify_variant(question, v)

        if result == "accept":
            acc.append(v)
        elif result == "review":
            rev.append(v)
        else:
            rej.append(v)

    accepted.append({
        "question": question,
        "answer": answer,
        "variants": acc
    })

    review.append({
        "question": question,
        "variants": rev
    })

    rejected.append({
        "question": question,
        "variants": rej
    })


# ===============================
# حفظ النتائج
# ===============================

with open(f"{OUTPUT_DIR}/accepted.json", "w", encoding="utf-8") as f:
    json.dump(accepted, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_DIR}/review.json", "w", encoding="utf-8") as f:
    json.dump(review, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_DIR}/rejected.json", "w", encoding="utf-8") as f:
    json.dump(rejected, f, ensure_ascii=False, indent=2)

print("✅ Done!")
print("✔ accepted.json")
print("⚠ review.json")
print("❌ rejected.json")