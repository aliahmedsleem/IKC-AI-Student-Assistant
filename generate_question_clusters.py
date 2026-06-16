# generate_question_clusters.py
# -*- coding: utf-8 -*-
"""
Generate safe question paraphrase clusters from the existing faqs.db without modifying
the original database. Output is a review file (JSON + CSV) that keeps every generated
variation linked to its original FAQ id and answer.

Design goals:
- Zero destructive changes to the current project
- Keep original questions and answers as source of truth
- Generate 3-5 paraphrases per question
- Prefer local MT5 if available; otherwise use rule-based templates
- Filter duplicates and unsafe / over-general variations

Usage:
    python generate_question_clusters.py
    python generate_question_clusters.py --db faqs.db --outdir generated_clusters
    python generate_question_clusters.py --use-mt5 --max-new-tokens 64

Outputs:
    generated_clusters/question_clusters_review.json
    generated_clusters/question_clusters_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------
# Text normalization utilities
# -----------------------------

ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652]")
NON_WORD_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "هل", "ما", "ماذا", "من", "منو", "شنو", "شلون", "وين", "ليش", "كم", "شكد",
    "في", "على", "عن", "الى", "إلى", "مع", "بين", "هذا", "هذه", "ذلك", "تلك",
    "هناك", "هنا", "هو", "هي", "هم", "انا", "اني", "نحن", "انت", "انتي", "انتم",
    "او", "أو", "ثم", "لكن", "بس", "اذا", "إذا", "كل", "فقط", "ايضا", "أيضا",
    "يوجد", "اكو", "موجود", "موجودة", "ال", "و", "يا", "لو", "راح", "هسه",
    "بعد", "قبل", "داخل", "خارج", "حول", "مثل", "ابي", "أريد", "اريد"
}

GENERAL_BAD_VARIANTS = {
    # Overly broad one-word or low-value variants we do NOT want to add automatically
    "القانون", "الهندسه", "القبول", "التسجيل", "المسائي", "الموازي", "النقل",
    "قانون", "هندسه", "قبول", "تسجيل", "مسائي", "موازي", "نقل"
}

PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ب", "و", "ل")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("ـ", "")
    text = NON_WORD_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_token(tok: str) -> str:
    tok = normalize_text(tok)
    if tok.startswith("ال") and len(tok) > 3:
        tok = tok[2:]
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if tok.startswith(prefix) and len(tok) > len(prefix) + 2:
                tok = tok[len(prefix):]
                changed = True
                break
    return tok


def tokenize(text: str) -> List[str]:
    toks: List[str] = []
    for raw in normalize_text(text).split():
        tok = normalize_token(raw)
        if len(tok) <= 1:
            continue
        if tok in STOPWORDS:
            continue
        toks.append(tok)
    return toks


# -----------------------------
# Data model
# -----------------------------

@dataclass
class FAQ:
    faq_id: int
    question: str
    answer: str
    category: str


# -----------------------------
# Load data
# -----------------------------

def load_faqs(db_path: Path) -> List[FAQ]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, question, answer, COALESCE(category,'') FROM faqs ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [FAQ(int(r[0]), r[1] or "", r[2] or "", r[3] or "") for r in rows]


# -----------------------------
# Heuristic paraphrase builder
# -----------------------------

def question_type(question: str) -> str:
    q = normalize_text(question)
    if "معدل" in q or "الحد الادنى" in q or "قبول" in q:
        return "admission"
    if "اجور" in q or "رسوم" in q or "قسط" in q:
        return "fees"
    if "نقل" in q or "باصات" in q or "خطوط" in q:
        return "transport"
    if "موقع" in q or "عنوان" in q:
        return "location"
    if "تقديم" in q or "التقديم" in q:
        return "application"
    if "مسائي" in q or "صباحي" in q or "موازي" in q:
        return "study_mode"
    if "سكن" in q:
        return "housing"
    if "تخفيض" in q:
        return "discount"
    return "general"


def find_named_entities(question: str) -> Dict[str, str]:
    q = normalize_text(question)
    entities: Dict[str, str] = {}

    # Sections / departments
    if "هندسه تقنيات الحاسوب" in q or ("هندسه" in q and "تقنيات" in q and "حاسوب" in q):
        entities["dept"] = "هندسه تقنيات الحاسوب"
    elif "القانون" in q:
        entities["dept"] = "القانون"

    # constraints / qualifiers
    if "مسائي" in q:
        entities["mode"] = "مسائي"
    elif "صباحي" in q:
        entities["mode"] = "صباحي"
    elif "موازي" in q:
        entities["mode"] = "موازي"

    if "بغداد" in q:
        entities["place"] = "بغداد"

    if "ذوي الاحتياجات" in q:
        entities["group"] = "ذوي الاحتياجات الخاصه"

    return entities


def make_rule_based_variants(question: str) -> List[str]:
    q_norm = normalize_text(question)
    q_type = question_type(question)
    ent = find_named_entities(question)
    dept = ent.get("dept", "")
    mode = ent.get("mode", "")
    place = ent.get("place", "")
    group = ent.get("group", "")

    variants: List[str] = []

    if q_type == "admission":
        if dept:
            variants.extend([
                f"شكد معدل {dept}",
                f"كم القبول ب{dept}",
                f"شنو الحد الادنى ل{dept}",
                f"معدل القبول ب{dept} شكد",
            ])
        else:
            variants.extend([
                "شكد القبول",
                "كم معدل القبول",
                "شنو الحد الادنى للقبول",
            ])

    elif q_type == "fees":
        if dept and mode:
            variants.extend([
                f"شكد اجور {dept} {mode}",
                f"كم رسوم {dept} {mode}",
                f"القسط مال {dept} {mode} شكد",
                f"شكد اقساط {dept} {mode}",
            ])
        elif dept:
            variants.extend([
                f"شكد اجور {dept}",
                f"كم رسوم {dept}",
                f"القسط مال {dept} شكد",
            ])

    elif q_type == "transport":
        variants.extend([
            "اكو نقل للكلية",
            "هل توجد خطوط نقل",
            "اكو باصات للطلاب",
            "اريد خط نقل من والى الجامعه",
        ])

    elif q_type == "location":
        if place:
            variants.extend([
                f"وين موقع الكليه ب{place}",
                f"عنوان الكليه ب{place} وين",
                f"وين الكليه الرئيسيه",
            ])
        else:
            variants.extend([
                "وين موقع الكليه",
                "عنوان الكليه وين",
            ])

    elif q_type == "application":
        variants.extend([
            "شلون اقدم عالكليه",
            "طريقة التقديم شنو",
            "كيف اقدم على الكليه",
            "التسجيل والتقديم شلون يصير",
        ])

    elif q_type == "study_mode":
        if mode == "موازي":
            if dept:
                variants.extend([
                    f"اكو موازي ب{dept}",
                    f"هل يوجد قبول للموازي في {dept}",
                    f"الموازي ب{dept} موجود",
                ])
            else:
                variants.extend([
                    "اكو نظام موازي",
                    "هل الكليه بيها موازي",
                    "الموازي موجود لو لا",
                ])
        elif mode == "مسائي":
            if dept:
                variants.extend([
                    f"اكو {dept} مسائي",
                    f"هل يمكن دراسة {dept} مسائي",
                    f"الدوام المسائي ب{dept} موجود",
                ])
            else:
                variants.extend([
                    "اكو دراسة مسائيه",
                    "هل الكليه بيها مسائي",
                    "المسائي موجود لو لا",
                ])
        elif mode == "صباحي":
            if dept:
                variants.extend([
                    f"اكو {dept} صباحي",
                    f"هل توجد دراسة صباحيه في {dept}",
                ])

    elif q_type == "housing":
        variants.extend([
            "اكو سكن للطلاب",
            "هل توجد اقسام داخليه",
            "يوجد سكن للطلبه",
        ])

    elif q_type == "discount":
        if group:
            variants.extend([
                "شكد التخفيض لذوي الاحتياجات",
                "نسبة التخفيض لذوي الاحتياجات شكد",
                "اكو تخفيض لذوي الاحتياجات",
            ])

    else:
        # Gentle generic variants only; avoid overly broad paraphrases
        toks = tokenize(question)
        if len(toks) >= 3:
            variants.extend([
                " ".join(toks),
                question.replace("هل ", "").replace("ما هو ", "").replace("ما هي ", "").strip(" ؟?"),
            ])

    # Always add a slightly shortened normalized version if still informative
    toks = tokenize(question)
    if 2 <= len(toks) <= 6:
        variants.append(" ".join(toks))

    return variants


# -----------------------------
# Optional local MT5 generator
# -----------------------------

def try_load_mt5(use_mt5: bool):
    if not use_mt5:
        return None
    try:
        from transformers import pipeline
        return pipeline(
            "text2text-generation",
            model="google/mt5-small",
            tokenizer="google/mt5-small",
        )
    except Exception as exc:
        print(f"[WARN] Could not load MT5. Falling back to rule-based only. Details: {exc}")
        return None


def mt5_variants(pipe, question: str, max_new_tokens: int = 64) -> List[str]:
    if pipe is None:
        return []

    prompt = (
        "اعط 5 صياغات عربية مختلفة لنفس السؤال التالي بدون تغيير المعنى، "
        "بعضها عامي وبعضها فصيح، وكل صياغة في سطر مستقل:\n"
        f"{question}"
    )

    try:
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = out[0]["generated_text"].strip()
    except Exception as exc:
        print(f"[WARN] MT5 generation failed for question: {question}\n{exc}")
        return []

    lines = []
    for line in re.split(r"[\n\r]+", text):
        line = line.strip("-•* \t")
        if line:
            lines.append(line)
    return lines


# -----------------------------
# Filtering
# -----------------------------

def is_too_broad(variant: str) -> bool:
    norm = normalize_text(variant)
    if not norm:
        return True
    if norm in GENERAL_BAD_VARIANTS:
        return True
    toks = tokenize(variant)
    if len(toks) == 0:
        return True
    if len(toks) == 1 and toks[0] in GENERAL_BAD_VARIANTS:
        return True
    return False


def semantic_guard(original_question: str, variant: str) -> bool:
    """
    Conservative safety gate:
    - variant must keep at least one anchor/department/important token from the original
    - prevent shrinking into a very broad phrase
    """
    orig_tokens = set(tokenize(original_question))
    var_tokens = set(tokenize(variant))

    if not var_tokens:
        return False

    shared = orig_tokens.intersection(var_tokens)
    if len(shared) == 0:
        return False

    # Keep department or important domain terms when present in original
    protected_terms = {
        "قانون", "هندسه", "تقنيات", "حاسوب", "موازي", "مسائي", "صباحي",
        "نقل", "باصات", "خطوط", "قبول", "معدل", "اجور", "رسوم", "سكن",
        "تقديم", "موقع", "بغداد", "ذوي", "احتياجات"
    }
    protected_in_original = orig_tokens.intersection(protected_terms)
    if protected_in_original and not var_tokens.intersection(protected_in_original):
        return False

    # Avoid generating something much broader than the source
    if len(var_tokens) < 2 and len(orig_tokens) >= 3:
        return False

    return True


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        norm = normalize_text(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
    return out


def build_variants(question: str, mt5_pipe=None, max_new_tokens: int = 64, target_n: int = 4) -> List[str]:
    candidates = []

    # Rule-based first: stable and safe
    candidates.extend(make_rule_based_variants(question))

    # MT5 next: richer paraphrases if available
    candidates.extend(mt5_variants(mt5_pipe, question, max_new_tokens=max_new_tokens))

    # Cleanup
    cleaned: List[str] = []
    for item in dedupe_preserve_order(candidates):
        item = item.strip(" ؟?").strip()
        if normalize_text(item) == normalize_text(question):
            continue
        if is_too_broad(item):
            continue
        if not semantic_guard(question, item):
            continue
        cleaned.append(item)

    # Trim to requested count
    return cleaned[:target_n]


# -----------------------------
# Main
# -----------------------------

def export_review_files(faqs: List[FAQ], outdir: Path, mt5_pipe=None, max_new_tokens: int = 64, target_n: int = 4) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "question_clusters_review.json"
    csv_path = outdir / "question_clusters_review.csv"

    json_payload: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []

    for faq in faqs:
        variants = build_variants(
            faq.question,
            mt5_pipe=mt5_pipe,
            max_new_tokens=max_new_tokens,
            target_n=target_n,
        )

        item = {
            "faq_id": faq.faq_id,
            "category": faq.category,
            "canonical_question": faq.question,
            "answer": faq.answer,
            "generated_variants": variants,
        }
        json_payload.append(item)

        for idx, variant in enumerate(variants, start=1):
            csv_rows.append({
                "faq_id": faq.faq_id,
                "category": faq.category,
                "canonical_question": faq.question,
                "variant_index": idx,
                "generated_variant": variant,
                "answer": faq.answer,
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "faq_id", "category", "canonical_question",
                "variant_index", "generated_variant", "answer"
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="faqs.db", help="Path to SQLite database")
    parser.add_argument("--outdir", default="generated_clusters", help="Output directory")
    parser.add_argument("--use-mt5", action="store_true", help="Use local google/mt5-small if available")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--target-n", type=int, default=4, help="Target variants per question")
    args = parser.parse_args()

    db_path = Path(args.db)
    outdir = Path(args.outdir)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    faqs = load_faqs(db_path)
    print(f"[INFO] Loaded {len(faqs)} FAQs from {db_path}")

    mt5_pipe = try_load_mt5(args.use_mt5)
    if mt5_pipe is not None:
        print("[INFO] MT5 loaded successfully.")
    else:
        print("[INFO] Using rule-based generation only.")

    json_path, csv_path = export_review_files(
        faqs=faqs,
        outdir=outdir,
        mt5_pipe=mt5_pipe,
        max_new_tokens=args.max_new_tokens,
        target_n=args.target_n,
    )

    print(f"[OK] JSON review file: {json_path}")
    print(f"[OK] CSV review file: {csv_path}")
    print("[DONE] No changes were made to the original database.")


if __name__ == "__main__":
    main()
