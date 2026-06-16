# rag_engine.py
# -*- coding: utf-8 -*-

import os
import re
import math
import sqlite3
from typing import List, Dict, Any, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "faqs.db"
CHROMA_DIR = "chroma_faqs"
COLLECTION = "faqs"

# ===== Embedding settings =====
EMBEDDING_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_LOCAL_DIR = os.getenv(
    "HF_EMBEDDING_LOCAL_DIR",
    os.path.join("models", "BAAI__bge-m3")
)
HF_OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

NO_MATCH_MESSAGE = (
    "سؤالك لم يطابق أيًا من بياناتنا، يستحسن أن تزيد من تفاصيل سؤالك "
    "ليتسنى لنا خدمتك بصورة أفضل ... جميع الاستفسارات مهمة لنا ونأخذ "
    "كل رسالة من حضراتكم بجدية تامة."
)

SHORT_SUGGESTION_PREFIX = "هل تقصد السؤال عن:"


# =========================================================
# Normalization
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()

    # Arabic normalization
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    # remove tashkeel
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[ـ]+", "", text)

    # digits
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    text = text.translate(str.maketrans(arabic_digits, western_digits))

    # common variants
    replacements = {
        "اون لاين": "اونلاين",
        "على النت": "اونلاين",
        "الكترونيا": "الكتروني",
        "إلكترونيا": "الكتروني",
        "إلكتروني": "الكتروني",
        "الالكتروني": "الكتروني",
        "انترنت": "اونلاين",
        "الإنترنت": "اونلاين",
        "مسائية": "مسائي",
        "صباحية": "صباحي",
        "طالبه": "طالبات",
        "طالبهً": "طالبات",
        "الخريج": "خريج",
        "للخريجين": "خريج",
        "خريجي": "خريج",
        "الخريجين": "خريج",
        "الوظائف": "وظائف",
        "مجال العمل": "مجالات العمل",
        "فرص الشغل": "فرص العمل",
        "الشغل": "العمل",
        "اقسام": "اقسام",
        "القسم": "قسم",
        "التخصص": "قسم",
        "تخصص": "قسم",
        "الانكليزي": "انكليزي",
        "الانجليزية": "انكليزي",
        "الانكليزيه": "انكليزي",
        "العربية": "عربيه",
        "العربي": "عربيه",
        "العربيه": "عربيه",
        "السيبرانيه": "سيبراني",
        "السيبراني": "سيبراني",
        "المصرفية": "مصرفيه",
        "المالية": "ماليه",
        "الاعمال": "اعمال",
        "الإعلام": "اعلام",
        "العلوم السياسيه": "علوم سياسيه",
        "السياسية": "سياسيه",
        "رياض الأطفال": "رياض الاطفال",
        "القرآن": "قران",
        "الشريعة": "شريعه",
        "الاداره": "اداره",
        "اسجل": "تسجيل",
        "اسجل؟": "تسجيل",
        "التسجيل": "تسجيل",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_AR_STOPWORDS = {
    normalize_text(x) for x in {
        "هل", "ما", "ماذا", "من", "منو", "شنو", "شلون", "وين", "ليش", "كم",
        "في", "على", "عن", "الى", "إلى", "منه", "منها", "مع", "بين",
        "هذا", "هذه", "ذلك", "تلك", "هناك", "هنا", "هو", "هي", "هم",
        "انا", "اني", "نحن", "انت", "انتي", "انتم", "أو", "او", "ثم",
        "لكن", "بس", "اذا", "إذا", "كل", "فقط", "ايضا", "أيضا",
        "يوجد", "اكو", "عدكم", "عندكم", "موجود", "موجوده", "موجودة",
        "ال", "و", "يا", "لو", "راح", "هسه", "بعد", "قبل", "داخل", "خارج",
        "حول", "مثل", "هوه", "هاي", "هذاك", "هذي", "شنهي", "شنوهي",
        "اريد", "أريد", "ابي", "هاي", "هذا", "هذي", "شي", "شكد", "كم",
        "اكدر", "أكدر", "ممكن", "يمكن", "صار", "صاير", "تدلني", "اعرف",
        "بشكل", "عام", "تفاصيل", "كامل", "كامله", "شرح", "معلومات", "توضيح"
    }
}

_RAW_DOMAIN_SYNONYMS = {
    "موازي": "نظام موازي",
    "مسائي": "دراسة مسائي دوام مسائي",
    "صباحي": "دراسة صباحي دوام صباحي",
    "اجور": "رسوم اقساط قسط تكلفة اجور",
    "اقساط": "رسوم اقساط قسط تكلفة اجور",
    "رسوم": "رسوم اقساط قسط تكلفة اجور",
    "تكلفه": "رسوم اقساط قسط تكلفة اجور",
    "تقسيط": "اقساط رسوم اجور تقسيط",
    "خطوط": "نقل باصات خطوط مواصلات",
    "باصات": "نقل باصات خطوط مواصلات",
    "نقل": "نقل باصات خطوط مواصلات",
    "رئيس": "رئيس القسم",
    "عميد": "عمادة الكلية العميد",
    "قبول": "القبول التسجيل معدل",
    "تسجيل": "التسجيل القبول تقديم اسجل",
    "اسجل": "التسجيل القبول تقديم اسجل",
    "تقديم": "التقديم التسجيل اونلاين الكتروني",
    "اونلاين": "اونلاين الكتروني تقديم تسجيل",
    "الكتروني": "اونلاين الكتروني تقديم تسجيل",
    "معدل": "الحد الادنى معدل القبول",
    "وظائف": "عمل وظائف مجالات العمل فرص العمل خريج مستقبل",
    "عمل": "عمل وظائف مجالات العمل فرص العمل خريج مستقبل",
    "خريج": "عمل وظائف مجالات العمل فرص العمل خريج مستقبل",
    "مستقبل": "عمل وظائف مجالات العمل فرص العمل خريج مستقبل",
    "مواد": "مواد المواد يدرس يدرسون منهج دراسي",
    "يدرس": "مواد المواد يدرس يدرسون منهج دراسي طبيعة الدراسة",
    "يدرسون": "مواد المواد يدرس يدرسون منهج دراسي طبيعة الدراسة",
    "دراسة": "دراسة طبيعة الدراسة نظري عملي",
    "نظري": "دراسة طبيعة الدراسة نظري عملي",
    "عملي": "دراسة طبيعة الدراسة نظري عملي",
    "سكن": "سكن اقسام داخلية طلبة محافظات",
    "عنوان": "عنوان موقع مكان",
    "موقع": "عنوان موقع مكان",
    "دوام": "دوام اوقات الدوام ساعات الدوام",
    "افضل": "افضل احسن",
    # departments
    "قران": "علوم القران الحديث",
    "حديث": "علوم القران الحديث",
    "شريعه": "شريعه",
    "فكر": "فكر اسلامي",
    "اسلامي": "فكر اسلامي",
    "عربيه": "لغة عربيه ادابها",
    "تاريخ": "تاريخ",
    "قانون": "قانون",
    "سياسيه": "علوم سياسيه",
    "اعلام": "اعلام",
    "انكليزي": "لغة انكليزي",
    "هندسه": "هندسه تقنيات الحاسوب",
    "حاسوب": "هندسه تقنيات الحاسوب حاسوب",
    "حاسبات": "هندسه تقنيات الحاسوب حاسوب",
    "ماليه": "علوم ماليه مصرفيه",
    "مصرفيه": "علوم ماليه مصرفيه",
    "رياض": "رياض الاطفال تربيه خاصه",
    "اطفال": "رياض الاطفال تربيه خاصه",
    "تربيه": "رياض الاطفال تربيه خاصه",
    "اعمال": "اداره الاعمال",
    "اداره": "اداره الاعمال",
    "سيبراني": "امن سيبراني",
    "امن": "امن سيبراني",
}

_DOMAIN_SYNONYMS = {
    normalize_text(k): normalize_text(v)
    for k, v in _RAW_DOMAIN_SYNONYMS.items()
}

_RAW_ANCHOR_TERMS = {
    "رئيس", "عميد", "مسؤول", "مقرر",
    "معدل", "حد", "ادنى", "الادنى",
    "اجور", "رسوم", "اقساط", "قسط", "تكلفه", "تقسيط",
    "موازي", "مسائي", "صباحي",
    "نقل", "باصات", "خطوط",
    "دوام", "تسجيل", "قبول", "تقديم", "اسجل",
    "ماستر", "ماجستير",
    "مواد", "المواد", "اسم", "اسماء",
    "عنوان", "نجف",
    "مقاعد", "بنات", "طالبات",
    "سكن", "عمل", "وظائف", "خريج",
}
_ANCHOR_TERMS = {normalize_text(x) for x in _RAW_ANCHOR_TERMS}

_RAW_STRICT_ANCHOR_FORMS = {
    "رئيس": ["رئيس", "رئيس القسم"],
    "عميد": ["عميد", "العميد", "عمادة الكلية"],
    "مسؤول": ["مسؤول"],
    "مقرر": ["مقرر"],
    "معدل": ["معدل", "الحد الادنى", "حد ادنى", "الادنى"],
    "اجور": ["اجور", "رسوم", "اقساط", "قسط", "تكلفه"],
    "رسوم": ["رسوم", "اجور", "اقساط", "قسط", "تكلفه"],
    "اقساط": ["اقساط", "رسوم", "اجور", "قسط", "تقسيط"],
    "قسط": ["قسط", "اقساط", "رسوم", "اجور", "تقسيط"],
    "تكلفه": ["تكلفه", "رسوم", "اجور"],
    "تقسيط": ["تقسيط", "اقساط", "قسط"],
    "موازي": ["موازي", "نظام موازي"],
    "مسائي": ["مسائي", "دراسة مسائي", "دوام مسائي"],
    "صباحي": ["صباحي", "دراسة صباحي", "دوام صباحي"],
    "نقل": ["نقل", "خطوط", "باصات", "خطوط نقل", "مواصلات"],
    "باصات": ["باصات", "نقل", "خطوط نقل"],
    "خطوط": ["خطوط", "نقل", "خطوط نقل"],
    "دوام": ["دوام", "اوقات الدوام", "ساعات الدوام"],
    "تسجيل": ["تسجيل", "القبول", "التسجيل", "تقديم", "اسجل"],
    "اسجل": ["تسجيل", "القبول", "التسجيل", "تقديم", "اسجل"],
    "تقديم": ["تقديم", "تسجيل", "الكتروني", "اونلاين"],
    "قبول": ["قبول", "القبول"],
    "ماستر": ["ماستر", "ماجستير", "دراسات عليا"],
    "ماجستير": ["ماستر", "ماجستير", "دراسات عليا"],
    "مواد": ["مواد", "المواد", "اسماء المواد", "اسم المواد"],
    "المواد": ["مواد", "المواد", "اسماء المواد", "اسم المواد"],
    "اسم": ["اسم", "اسماء"],
    "اسماء": ["اسماء", "اسم"],
    "عنوان": ["عنوان", "العنوان", "موقع"],
    "نجف": ["نجف", "النجف"],
    "مقاعد": ["مقاعد", "عدد المقاعد"],
    "بنات": ["بنات", "طالبات"],
    "طالبات": ["طالبات", "بنات"],
    "سكن": ["سكن", "اقسام داخلية", "طلبة المحافظات"],
    "عمل": ["عمل", "وظائف", "مجالات العمل", "فرص العمل", "خريج", "مستقبل"],
    "وظائف": ["عمل", "وظائف", "مجالات العمل", "فرص العمل", "خريج", "مستقبل"],
    "خريج": ["عمل", "وظائف", "مجالات العمل", "فرص العمل", "خريج", "مستقبل"],
}
STRICT_ANCHOR_FORMS = {
    normalize_text(k): [normalize_text(v) for v in vals]
    for k, vals in _RAW_STRICT_ANCHOR_FORMS.items()
}

_RAW_CONSTRAINT_TERMS = {
    "صباحي", "مسائي", "بنات", "طالبات", "ماستر", "ماجستير",
    "مواد", "المواد", "اسم", "اسماء", "عنوان", "نجف",
    "مقاعد", "مقرر", "رئيس", "عميد", "مسؤول", "عمل", "وظائف", "خريج"
}
_CONSTRAINT_TERMS = {normalize_text(x) for x in _RAW_CONSTRAINT_TERMS}

_INTENT_RULES = {
    "position": {"رئيس", "عميد", "مسؤول", "مقرر"},
    "materials": {"مواد", "المواد", "اسماء", "اسم", "يدرس", "يدرسون"},
    "address": {"عنوان", "نجف", "فرع", "موقع", "مكان", "اين", "وين"},
    "seats": {"مقاعد"},
    "postgrad": {"ماستر", "ماجستير", "دراسات", "عليا"},
    "transport": {"نقل", "باصات", "خطوط", "مواصلات"},
    "housing": {"سكن"},
    "fees": {"اجور", "رسوم", "اقساط", "قسط", "تكلفه", "تقسيط"},
    "admission": {"قبول", "معدل", "الادنى", "ادنى", "حد"},
    "application": {"تقديم", "تسجيل", "اونلاين", "الكتروني", "اسجل"},
    "study_time": {"صباحي", "مسائي", "دوام"},
    "female_only": {"بنات", "طالبات"},
    "career": {"عمل", "وظائف", "خريج", "مجالات", "مستقبل"},
    "study_nature": {"طبيعه", "نظري", "عملي", "دراسه"},
    "count": {"عدد", "كم", "اقسام"},
    "opinion": {"افضل", "احسن"},
    "comparison": {"فرق", "مقارنه", "بين"},
}

# Strong department aliases
_RAW_DEPARTMENT_ALIASES = {
    "علوم القران والحديث": ["علوم القران والحديث", "علوم القران", "القران والحديث", "القران", "الحديث"],
    "الشريعه": ["الشريعه", "قسم الشريعه"],
    "الفكر الاسلامي": ["الفكر الاسلامي", "فكر اسلامي"],
    "اللغه العربيه وادابها": ["اللغه العربيه وادابها", "اللغه العربيه", "العربيه", "ادابها"],
    "التاريخ": ["التاريخ", "قسم التاريخ"],
    "القانون": ["القانون", "قسم القانون"],
    "العلوم السياسيه": ["العلوم السياسيه", "علوم سياسيه", "السياسه", "السياسيه"],
    "الاعلام": ["الاعلام", "قسم الاعلام"],
    "اللغه الانكليزيه": ["اللغه الانكليزيه", "اللغه الانكليزي", "الانكليزيه", "الانكليزي"],
    "هندسه تقنيات الحاسوب": ["هندسه تقنيات الحاسوب", "هندسه الحاسوب", "تقنيات الحاسوب", "هندسه تقنيات", "حاسوب", "حاسبات"],
    "العلوم الماليه والمصرفيه": ["العلوم الماليه والمصرفيه", "العلوم الماليه", "المصرفيه", "ماليه ومصرفيه"],
    "رياض الاطفال والتربيه الخاصه": ["رياض الاطفال والتربيه الخاصه", "رياض الاطفال", "التربيه الخاصه"],
    "اداره الاعمال": ["اداره الاعمال", "الاعمال", "اداره"],
    "الامن السيبراني": ["الامن السيبراني", "امن سيبراني", "سيبراني"],
}
DEPARTMENT_ALIASES = {
    normalize_text(k): [normalize_text(v) for v in vals]
    for k, vals in _RAW_DEPARTMENT_ALIASES.items()
}


# =========================================================
# Token helpers
# =========================================================
def normalize_token(tok: str) -> str:
    tok = normalize_text(tok)

    if not tok:
        return tok

    if tok.startswith("ال") and len(tok) > 3:
        tok = tok[2:]

    prefixes = ("وال", "بال", "كال", "فال", "لل", "ب", "و", "ل")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if tok.startswith(prefix) and len(tok) > len(prefix) + 2:
                tok = tok[len(prefix):]
                changed = True
                break

    return tok


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = []

    for raw_tok in text.split():
        tok = normalize_token(raw_tok)

        if len(tok) <= 1:
            continue
        if tok in _AR_STOPWORDS:
            continue

        tokens.append(tok)

    return tokens


def original_terms(text: str) -> List[str]:
    return list(dict.fromkeys(tokenize(text)))


def expand_query_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for tok in tokens:
        if tok in _DOMAIN_SYNONYMS:
            expanded.extend(tokenize(_DOMAIN_SYNONYMS[tok]))
    return list(dict.fromkeys(expanded))


def important_terms(text: str) -> List[str]:
    return expand_query_tokens(original_terms(text))


def safe_ratio(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


# =========================================================
# Intent / anchors / constraints / departments
# =========================================================
def extract_anchor_terms(text: str) -> List[str]:
    toks = original_terms(text)
    anchors = [t for t in toks if t in _ANCHOR_TERMS]
    return list(dict.fromkeys(anchors))


def extract_constraint_terms(text: str) -> List[str]:
    toks = original_terms(text)
    terms = [t for t in toks if t in _CONSTRAINT_TERMS]
    return list(dict.fromkeys(terms))


def extract_intents(text: str) -> List[str]:
    toks = set(important_terms(text))
    found = []
    for intent_name, keywords in _INTENT_RULES.items():
        norm_keys = {normalize_text(k) for k in keywords}
        if toks.intersection(norm_keys):
            found.append(intent_name)
    return found


def extract_departments(text: str) -> List[str]:
    text_norm = normalize_text(text)
    found = []
    for dept, aliases in DEPARTMENT_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in text_norm:
                found.append(dept)
                break
    return list(dict.fromkeys(found))


def department_match_ratio(query: str, candidate_text: str) -> Tuple[float, List[str]]:
    q_depts = extract_departments(query)
    if not q_depts:
        return 1.0, []

    c_depts = extract_departments(candidate_text)
    matched = [d for d in q_depts if d in c_depts]
    return safe_ratio(len(matched), len(q_depts)), matched


def has_intent_mismatch(query: str, candidate_text: str) -> bool:
    q_intents = set(extract_intents(query))
    if not q_intents:
        return False

    c_intents = set(extract_intents(candidate_text))

    # allow close pairs
    if "application" in q_intents and "admission" in c_intents:
        q_intents = q_intents - {"application"}
    if "admission" in q_intents and "application" in c_intents:
        c_intents = c_intents | {"admission"}

    # career should not match study nature/materials
    if "career" in q_intents and "career" not in c_intents:
        return True
    if "materials" in q_intents and "materials" not in c_intents:
        if "study_nature" in c_intents:
            return True

    # opinion/count/comparison style questions usually no direct faq answer unless explicit
    if "opinion" in q_intents and "opinion" not in c_intents:
        return True
    if "count" in q_intents and "count" not in c_intents:
        return True
    if "comparison" in q_intents and "comparison" not in c_intents:
        return True

    return not q_intents.issubset(c_intents)


def candidate_contains_anchor(candidate_text: str, anchor: str) -> bool:
    cand_norm = normalize_text(candidate_text)
    cand_terms = set(tokenize(candidate_text))

    if anchor in cand_terms:
        return True

    strict_forms = STRICT_ANCHOR_FORMS.get(anchor, [anchor])

    for form in strict_forms:
        form_norm = normalize_text(form)
        if " " not in form_norm:
            if normalize_token(form_norm) in cand_terms:
                return True
        else:
            if form_norm in cand_norm:
                return True

    return False


def anchor_match_ratio(query: str, candidate_text: str) -> Tuple[float, List[str]]:
    anchors = extract_anchor_terms(query)
    if not anchors:
        return 1.0, []

    matched = []
    for anchor in anchors:
        if candidate_contains_anchor(candidate_text, anchor):
            matched.append(anchor)

    return safe_ratio(len(matched), len(anchors)), matched


def constraint_match_ratio(query: str, candidate_text: str) -> Tuple[float, List[str]]:
    constraints = extract_constraint_terms(query)
    if not constraints:
        return 1.0, []

    matched = []
    for term in constraints:
        if candidate_contains_anchor(candidate_text, term):
            matched.append(term)

    return safe_ratio(len(matched), len(constraints)), matched


# =========================================================
# FAQ helpers
# =========================================================
def split_paraphrases(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("||")]
    return [p for p in parts if p]


def exact_match_any(question: str, candidates: List[str]) -> bool:
    qn = normalize_text(question)
    if not qn:
        return False
    for c in candidates:
        if normalize_text(c) == qn:
            return True
    return False


def overlap_stats(query: str, candidate_text: str) -> Tuple[int, float, List[str], int]:
    q_terms = set(important_terms(query))
    c_terms = set(important_terms(candidate_text))

    if not q_terms:
        return 0, 0.0, [], 0

    matched = sorted(q_terms.intersection(c_terms))
    overlap_count = len(matched)
    coverage = safe_ratio(overlap_count, len(q_terms))
    return overlap_count, coverage, matched, len(q_terms)


def is_short_query(question: str) -> bool:
    return len(original_terms(question)) <= 3


def is_near_exact(question: str, candidate_question: str) -> bool:
    qn = normalize_text(question)
    cn = normalize_text(candidate_question)
    return bool(qn and cn and qn == cn)


# =========================================================
# Vector DB
# =========================================================
def _get_embedding_source() -> str:
    if os.path.isdir(EMBEDDING_LOCAL_DIR):
        print(f"✅ Using local embedding model: {EMBEDDING_LOCAL_DIR}")
        return EMBEDDING_LOCAL_DIR

    print(f"🌐 Using remote embedding model: {EMBEDDING_MODEL_NAME}")
    return EMBEDDING_MODEL_NAME


def _build_embeddings():
    model_source = _get_embedding_source()

    if HF_OFFLINE_MODE and not os.path.isdir(EMBEDDING_LOCAL_DIR):
        raise RuntimeError(
            f"HF_HUB_OFFLINE=1 but local embedding model not found at: {EMBEDDING_LOCAL_DIR}"
        )

    return HuggingFaceEmbeddings(
        model_name=model_source,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectordb():
    if not os.path.exists(CHROMA_DIR):
        print("⚠️ chroma_faqs not found. SQLite mode only.")
        return None

    try:
        embeddings = _build_embeddings()
        return Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION,
            embedding_function=embeddings,
        )
    except Exception as e:
        print("⚠️ Failed to load Chroma/Embeddings, fallback to SQLite only:", e)
        return None


# =========================================================
# SQLite loading
# =========================================================
def fetch_all_faqs() -> List[Tuple[int, str, str, str, str]]:
    if not os.path.exists(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("PRAGMA table_info(faqs)")
        cols = [row[1] for row in c.fetchall()]
        has_paraphrase = "paraphrase" in cols

        if has_paraphrase:
            c.execute("SELECT id, question, answer, COALESCE(category,''), COALESCE(paraphrase,'') FROM faqs")
        else:
            c.execute("SELECT id, question, answer, COALESCE(category,''), '' FROM faqs")

        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print("⚠️ SQLite read error:", e)
        return []


# =========================================================
# Candidate builders
# =========================================================
def build_sqlite_candidates(question: str) -> List[Dict[str, Any]]:
    rows = fetch_all_faqs()
    candidates = []

    for faq_id, q, a, cat, paraphrase in rows:
        q = (q or "").strip()
        a = (a or "").strip()
        cat = (cat or "").strip()
        paraphrase = (paraphrase or "").strip()

        para_list = split_paraphrases(paraphrase)
        blob = f"{q} {paraphrase} {a} {cat}"

        overlap_count, coverage, matched_terms, query_len = overlap_stats(question, blob)
        anchor_ratio, matched_anchors = anchor_match_ratio(question, blob)
        constraint_ratio, matched_constraints = constraint_match_ratio(question, blob)
        dept_ratio, matched_departments = department_match_ratio(question, blob)

        q_norm = normalize_text(question)

        exact_bonus = 0.0
        if q_norm == normalize_text(q):
            exact_bonus = 1.0
        elif exact_match_any(question, para_list):
            exact_bonus = 0.95
        elif q_norm and q_norm in normalize_text(blob):
            exact_bonus = 0.2

        intent_penalty = 0.0
        if has_intent_mismatch(question, blob):
            intent_penalty = 0.60

        dept_penalty = 0.0
        if extract_departments(question) and dept_ratio < 1.0:
            dept_penalty = 0.65

        score = (
            (coverage * 0.55) +
            (safe_ratio(overlap_count, max(2, query_len)) * 0.15) +
            (anchor_ratio * 0.20) +
            (constraint_ratio * 0.20) +
            (dept_ratio * 0.70) +
            exact_bonus -
            intent_penalty -
            dept_penalty
        )

        candidates.append({
            "faq_id": int(faq_id),
            "question": q,
            "answer": a,
            "category": cat,
            "paraphrase": paraphrase,
            "paraphrase_list": para_list,
            "blob": blob,
            "overlap_count": overlap_count,
            "coverage": coverage,
            "matched_terms": matched_terms,
            "query_len": query_len,
            "anchor_ratio": anchor_ratio,
            "matched_anchors": matched_anchors,
            "constraint_ratio": constraint_ratio,
            "matched_constraints": matched_constraints,
            "department_ratio": dept_ratio,
            "matched_departments": matched_departments,
            "intent_mismatch": has_intent_mismatch(question, blob),
            "semantic_score": 0.0,
            "score": score,
            "source": "sqlite",
        })

    candidates.sort(
        key=lambda x: (
            -x["score"],
            x["intent_mismatch"],
            -x["department_ratio"],
            -x["constraint_ratio"],
            -x["anchor_ratio"],
            -x["coverage"],
            -x["overlap_count"],
        )
    )
    return candidates


def build_vector_candidates(question: str, vectordb, top_k: int = 10) -> List[Dict[str, Any]]:
    if vectordb is None:
        return []

    try:
        results = vectordb.similarity_search_with_score(question, k=top_k)
    except Exception as e:
        print("⚠️ Vector search failed:", e)
        return []

    candidates = []
    for doc, distance in results:
        q = (doc.metadata.get("question") or "").strip()
        a = (doc.metadata.get("answer") or "").strip()
        cat = (doc.metadata.get("category") or "").strip()
        faq_id = doc.metadata.get("faq_id")
        paraphrase = (doc.metadata.get("paraphrase") or "").strip()
        para_list = split_paraphrases(paraphrase)
        fulltext = (doc.page_content or "").strip()

        blob = f"{q} {paraphrase} {a} {cat} {fulltext}"
        overlap_count, coverage, matched_terms, query_len = overlap_stats(question, blob)
        anchor_ratio, matched_anchors = anchor_match_ratio(question, blob)
        constraint_ratio, matched_constraints = constraint_match_ratio(question, blob)
        dept_ratio, matched_departments = department_match_ratio(question, blob)

        semantic_score = 1.0 / (1.0 + max(float(distance or 0.0), 0.0))

        exact_bonus = 0.0
        if is_near_exact(question, q):
            exact_bonus = 0.4
        elif exact_match_any(question, para_list):
            exact_bonus = 0.35

        intent_penalty = 0.0
        if has_intent_mismatch(question, blob):
            intent_penalty = 0.45

        dept_penalty = 0.0
        if extract_departments(question) and dept_ratio < 1.0:
            dept_penalty = 0.55

        score = (
            (semantic_score * 0.40) +
            (coverage * 0.25) +
            (anchor_ratio * 0.15) +
            (constraint_ratio * 0.15) +
            (dept_ratio * 0.60) +
            exact_bonus -
            intent_penalty -
            dept_penalty
        )

        candidates.append({
            "faq_id": faq_id,
            "question": q,
            "answer": a,
            "category": cat,
            "paraphrase": paraphrase,
            "paraphrase_list": para_list,
            "blob": blob,
            "overlap_count": overlap_count,
            "coverage": coverage,
            "matched_terms": matched_terms,
            "query_len": query_len,
            "anchor_ratio": anchor_ratio,
            "matched_anchors": matched_anchors,
            "constraint_ratio": constraint_ratio,
            "matched_constraints": matched_constraints,
            "department_ratio": dept_ratio,
            "matched_departments": matched_departments,
            "intent_mismatch": has_intent_mismatch(question, blob),
            "semantic_score": semantic_score,
            "score": score,
            "source": "vector",
        })

    candidates.sort(
        key=lambda x: (
            -x["score"],
            x["intent_mismatch"],
            -x["department_ratio"],
            -x["constraint_ratio"],
            -x["anchor_ratio"],
            -x["coverage"],
            -x["overlap_count"],
        )
    )
    return candidates


def merge_candidates(question: str, vectordb) -> List[Dict[str, Any]]:
    vector_candidates = build_vector_candidates(question, vectordb, top_k=10)
    sqlite_candidates = build_sqlite_candidates(question)

    merged: Dict[str, Dict[str, Any]] = {}

    def add_candidate(item: Dict[str, Any]):
        key = normalize_text(item["question"])
        if not key:
            return
        if key not in merged:
            merged[key] = item
            return

        old = merged[key]
        if (
            item["score"] > old["score"] or
            (item["intent_mismatch"] is False and old["intent_mismatch"] is True) or
            item["department_ratio"] > old["department_ratio"] or
            item["constraint_ratio"] > old["constraint_ratio"] or
            item["anchor_ratio"] > old["anchor_ratio"] or
            item["coverage"] > old["coverage"]
        ):
            merged[key] = item

    for item in vector_candidates:
        add_candidate(item)

    for item in sqlite_candidates:
        add_candidate(item)

    final_list = list(merged.values())
    final_list.sort(
        key=lambda x: (
            -x["score"],
            x["intent_mismatch"],
            -x["department_ratio"],
            -x["constraint_ratio"],
            -x["anchor_ratio"],
            -x["coverage"],
            -x["overlap_count"],
        )
    )
    return final_list


# =========================================================
# Safety filters
# =========================================================
def candidate_is_too_specific_for_query(question: str, candidate_question: str, paraphrase_list: List[str]) -> bool:
    q_terms = set(original_terms(question))
    c_terms = set(original_terms(candidate_question))

    generic_terms = {
        "قسم", "قانون", "هندسه", "قبول", "معدل", "اجور", "رسوم", "مسائي",
        "موازي", "نقل", "كليه", "جامعه", "دراسه", "طالبات", "فروع",
        "عنوان", "موقع", "تقديم", "تسجيل", "سكن", "عمل", "وظائف", "خريج",
        "مواد", "تاريخ", "اعلام", "عربيه", "انكليزي", "سيبراني", "اعمال",
        "ماليه", "مصرفيه", "رياض", "اطفال", "شريعه", "قران", "سياسيه"
    }

    extra_specific = [t for t in c_terms - q_terms if t not in generic_terms]

    if len(q_terms) <= 4 and len(extra_specific) >= 2:
        return True

    return False


def has_explicit_conflict(query: str, candidate_text: str) -> bool:
    q = normalize_text(query)
    c = normalize_text(candidate_text)

    q_depts = extract_departments(q)
    c_depts = extract_departments(c)
    if q_depts and c_depts and not set(q_depts).intersection(set(c_depts)):
        return True

    # opinion question without explicit answer
    if any(x in q for x in ["افضل", "احسن"]) and not any(x in c for x in ["افضل", "احسن"]):
        return True

    # count question
    if ("عدد" in q and "اقسام" in q) and ("عدد" not in c and "اقسام" not in c):
        return True

    # general registration should not map to phone/contact only
    if any(x in q for x in ["تسجيل", "اسجل", "التقديم", "قبول"]) and any(x in c for x in ["رقم", "هاتف", "شؤون الطلبه", "شؤون الطلبة"]):
        return True

    # departments listing should not map to unrelated admin/info questions
    if ("اقسام" in q or "التخصصات" in q) and ("اقسام" not in c and "التخصصات" not in c):
        return True

    # comparison should not map to non-comparison faq
    if any(x in q for x in ["فرق", "مقارنه", "بين"]) and not any(x in c for x in ["فرق", "مقارنه", "بين"]):
        return True

    # صباحي vs مسائي
    if "صباحي" in q and "مسائي" in c:
        return True
    if "مسائي" in q and "صباحي" in c:
        return True

    # مواد vs فرص عمل
    if ("مواد" in q or "يدرس" in q or "يدرسون" in q) and any(x in c for x in ["فرص العمل", "مجالات العمل", "وظائف", "خريج"]):
        return True

    # فرص عمل vs مواد
    if any(x in q for x in ["عمل", "وظائف", "خريج", "مجالات", "مستقبل"]) and any(x in c for x in ["مواد", "المواد"]):
        return True

    # رئيس / عميد / مسؤول vs أي شيء آخر
    if any(x in q for x in ["رئيس", "عميد", "مسؤول", "مقرر"]) and not any(x in c for x in ["رئيس", "عميد", "مسؤول", "مقرر"]):
        return True

    # مقاعد vs معدل/قبول
    if "مقاعد" in q and "مقاعد" not in c:
        return True

    # عنوان فرع/نجف
    if ("عنوان" in q or "فرع" in q or "نجف" in q or "موقع" in q or "وين" in q) and ("عنوان" not in c and "فرع" not in c and "نجف" not in c and "موقع" not in c):
        if "اور" not in c and "بغداد" not in c:
            return True

    # نقل خاص للطالبات
    if ("نقل" in q and "طالبات" in q) and "نقل" not in c:
        return True

    # موازي صباحي
    if "موازي" in q and "صباحي" in q:
        if "موازي" not in c or "صباحي" not in c:
            return True

    # سكن
    if "سكن" in q and "سكن" not in c:
        return True

    # التسجيل shouldn't map to transfer
    if any(x in q for x in ["تسجيل", "اسجل", "التقديم", "قبول"]) and "انتقال" in c:
        return True

    # التقسيط shouldn't map to discounts unless explicit
    if "تقسيط" in q and "تقسيط" not in c:
        return True

    return False


# =========================================================
# Decision layer
# =========================================================
def strong_accept_for_long_question(best: Dict[str, Any], question: str) -> bool:
    anchors = extract_anchor_terms(question)
    constraints = extract_constraint_terms(question)
    q_terms = original_terms(question)
    q_depts = extract_departments(question)

    if best.get("intent_mismatch", False):
        return False

    if q_depts and best.get("department_ratio", 1.0) < 1.0:
        return False

    if exact_match_any(question, best.get("paraphrase_list", [])):
        return True

    if constraints and best["constraint_ratio"] < 1.0:
        return False

    if anchors and best["anchor_ratio"] < 1.0:
        return False

    if candidate_is_too_specific_for_query(question, best["question"], best.get("paraphrase_list", [])):
        return False

    if is_near_exact(question, best["question"]):
        return True

    min_overlap = max(2, math.ceil(len(q_terms) * 0.6))

    return (
        best["coverage"] >= 0.72 and
        best["overlap_count"] >= min_overlap and
        best["score"] >= 0.95
    )


def build_suggestions(question: str, candidates: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    anchors = extract_anchor_terms(question)
    constraints = extract_constraint_terms(question)
    q_depts = extract_departments(question)
    suggestions = []

    for item in candidates:
        if item.get("intent_mismatch", False):
            continue

        if has_explicit_conflict(question, item["blob"]):
            continue

        if constraints and item["constraint_ratio"] < 1.0:
            continue

        if anchors and item["anchor_ratio"] < 0.5:
            continue

        if q_depts and item.get("department_ratio", 1.0) < 1.0:
            continue

        q = item["question"].strip()
        if not q or q in suggestions:
            continue

        if (
            item["overlap_count"] >= 1 or
            item["coverage"] >= 0.20 or
            (anchors and item["anchor_ratio"] >= 0.5) or
            (constraints and item["constraint_ratio"] >= 0.5)
        ):
            suggestions.append(q)

        if len(suggestions) >= limit:
            break

    return suggestions


def short_query_decision(question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": None,
        }

    best = candidates[0]
    anchors = extract_anchor_terms(question)
    constraints = extract_constraint_terms(question)
    q_depts = extract_departments(question)
    q_intents = set(extract_intents(question))

    print("########## SHORT QUERY MODE ##########")
    print("🔎 Question:", question)
    print("🧾 Best Question:", best["question"])
    print("🗂 Paraphrases:", best.get("paraphrase_list", []))
    print("📌 Coverage:", round(best["coverage"], 4))
    print("🔑 Overlap:", best["overlap_count"])
    print("🎯 Anchor Ratio:", round(best["anchor_ratio"], 4))
    print("🎯 Matched Anchors:", best["matched_anchors"])
    print("🧱 Constraint Ratio:", round(best["constraint_ratio"], 4))
    print("🧱 Matched Constraints:", best["matched_constraints"])
    print("🏫 Department Ratio:", round(best.get("department_ratio", 1.0), 4))
    print("🏫 Matched Departments:", best.get("matched_departments", []))
    print("🧭 Intent mismatch:", best.get("intent_mismatch", False))
    print("⭐ Score:", round(best["score"], 4))
    print("🧩 Terms:", best["matched_terms"])
    print("🧠 Extracted Anchors:", anchors)
    print("🧠 Extracted Constraints:", constraints)
    print("🧠 Extracted Departments:", q_depts)
    print("-" * 70)

    if has_explicit_conflict(question, best["blob"]):
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    if best.get("intent_mismatch", False):
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    if q_depts and best.get("department_ratio", 1.0) < 1.0:
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    # generic unsupported questions should not force direct answers
    if any(x in q_intents for x in ["count", "opinion", "comparison"]):
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    short_strong_match = (
        is_near_exact(question, best["question"]) or
        exact_match_any(question, best.get("paraphrase_list", [])) or
        (
            best["coverage"] >= 0.50 and
            best["overlap_count"] >= 2 and
            (
                best["anchor_ratio"] >= 1.0 or
                len(best.get("paraphrase_list", [])) > 0
            )
        ) or
        (
            best["coverage"] >= 0.75 and best["overlap_count"] >= 1
        )
    )

    if short_strong_match and not (constraints and best["constraint_ratio"] < 1.0) and not (anchors and best["anchor_ratio"] < 0.5):
        return {
            "reply": best["answer"] or NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": best["faq_id"],
            "confidence": best["score"],
        }

    suggestion_pool = build_suggestions(question, candidates, limit=3)
    if suggestion_pool:
        reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
        return {
            "reply": reply,
            "suggestions": suggestion_pool,
            "matched_id": None,
            "confidence": best["score"],
        }

    return {
        "reply": NO_MATCH_MESSAGE,
        "suggestions": [],
        "matched_id": None,
        "confidence": best["score"],
    }


def long_query_decision(question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": None,
        }

    best = candidates[0]
    anchors = extract_anchor_terms(question)
    constraints = extract_constraint_terms(question)
    q_depts = extract_departments(question)
    q_intents = set(extract_intents(question))

    print("########## LONG QUERY MODE ##########")
    print("🔎 Question:", question)
    print("🧾 Best Question:", best["question"])
    print("🗂 Paraphrases:", best.get("paraphrase_list", []))
    print("📌 Coverage:", round(best["coverage"], 4))
    print("🔑 Overlap:", best["overlap_count"])
    print("🎯 Anchor Ratio:", round(best["anchor_ratio"], 4))
    print("🎯 Matched Anchors:", best["matched_anchors"])
    print("🧱 Constraint Ratio:", round(best["constraint_ratio"], 4))
    print("🧱 Matched Constraints:", best["matched_constraints"])
    print("🏫 Department Ratio:", round(best.get("department_ratio", 1.0), 4))
    print("🏫 Matched Departments:", best.get("matched_departments", []))
    print("🧭 Intent mismatch:", best.get("intent_mismatch", False))
    print("⭐ Score:", round(best["score"], 4))
    print("🧩 Terms:", best["matched_terms"])
    print("🧠 Extracted Anchors:", anchors)
    print("🧠 Extracted Constraints:", constraints)
    print("🧠 Extracted Departments:", q_depts)
    print("-" * 70)

    if has_explicit_conflict(question, best["blob"]):
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    if q_depts and best.get("department_ratio", 1.0) < 1.0:
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    # generic unsupported questions
    if "opinion" in q_intents or "count" in q_intents:
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    if strong_accept_for_long_question(best, question):
        return {
            "reply": best["answer"] or NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": best["faq_id"],
            "confidence": best["score"],
        }

    if (
        best.get("intent_mismatch", False) or
        (constraints and best["constraint_ratio"] < 1.0) or
        (anchors and best["anchor_ratio"] < 0.5)
    ):
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }

        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    if candidate_is_too_specific_for_query(question, best["question"], best.get("paraphrase_list", [])):
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }
        return {
            "reply": NO_MATCH_MESSAGE,
            "suggestions": [],
            "matched_id": None,
            "confidence": best["score"],
        }

    medium_match = (
        (best["coverage"] >= 0.45 and best["overlap_count"] >= 2 and best.get("department_ratio", 1.0) >= 1.0) or
        (best["score"] >= 1.05 and best["overlap_count"] >= 1 and best.get("department_ratio", 1.0) >= 1.0)
    )

    if medium_match:
        suggestion_pool = build_suggestions(question, candidates, limit=3)
        if suggestion_pool:
            reply = SHORT_SUGGESTION_PREFIX + "\n- " + "\n- ".join(suggestion_pool)
            return {
                "reply": reply,
                "suggestions": suggestion_pool,
                "matched_id": None,
                "confidence": best["score"],
            }

    return {
        "reply": NO_MATCH_MESSAGE,
        "suggestions": [],
        "matched_id": None,
        "confidence": best["score"],
    }


# =========================================================
# Public API
# =========================================================
def build_rag_chain():
    print("✅ RAG ENGINE FINAL MODE IS RUNNING")
    vectordb = load_vectordb()

    def answer(question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "reply": "الرجاء إدخال سؤال.",
                "suggestions": [],
                "matched_id": None,
                "confidence": None,
            }

        candidates = merge_candidates(question, vectordb)

        if is_short_query(question):
            return short_query_decision(question, candidates)

        return long_query_decision(question, candidates)

    return answer