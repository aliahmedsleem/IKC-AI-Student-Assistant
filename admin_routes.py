# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, request, redirect, url_for, Response
import sqlite3
import csv
import io
import json
from datetime import datetime
from typing import List, Dict
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_PATH = "faqs.db"
CLUSTERS_JSON = "question_clusters_review.json"
admin_bp = Blueprint("admin", __name__, template_folder="templates")

LOW_CONFIDENCE_THRESHOLD = 0.45


# lazy-loaded model globals
_TOKENIZER = None
_MODEL = None


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn, table_name: str, column_name: str) -> bool:
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in c.fetchall()]
    return column_name in cols


def ensure_schema():
    """
    يضمن وجود الأعمدة المطلوبة بدون كسر النظام إذا كانت غير موجودة.
    """
    conn = get_conn()
    c = conn.cursor()

    # paraphrase في faqs
    if not column_exists(conn, "faqs", "paraphrase"):
        c.execute("ALTER TABLE faqs ADD COLUMN paraphrase TEXT DEFAULT ''")

    # created_at في faqs
    if not column_exists(conn, "faqs", "created_at"):
        c.execute("ALTER TABLE faqs ADD COLUMN created_at TEXT")

    conn.commit()
    conn.close()


@admin_bp.before_app_request
def _prepare_schema():
    ensure_schema()


# =========================
# Paraphrase helpers
# =========================
def _fallback_paraphrases(question: str) -> List[str]:
    q = (question or '').strip()
    if not q:
        return []

    variants = []
    replacements = [
        ('وين', 'اين'),
        ('اكو', 'هل يوجد'),
        ('شلون', 'كيف'),
        ('شنو', 'ما هو'),
        ('عنوان', 'موقع'),
        ('موقع', 'عنوان'),
    ]

    for a, b in replacements:
        if a in q:
            variants.append(q.replace(a, b))

    variants.extend([
        f"اريد اعرف {q}",
        f"ممكن توضيح {q}",
        f"استفسار بخصوص {q}",
    ])

    clean = []
    seen = set()
    for item in variants:
        item = ' '.join((item or '').split()).strip()
        if not item:
            continue
        key = item.lower()
        if key == q.lower() or key in seen:
            continue
        seen.add(key)
        clean.append(item)
    return clean[:5]


def _load_generation_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        model_name = 'google/mt5-small'
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _TOKENIZER, _MODEL


def _clean_generated_text(text: str) -> str:
    text = ' '.join((text or '').split()).strip()
    if not text:
        return ''
    if '<extra_id_' in text:
        return ''
    return text


def generate_paraphrases_text(question: str) -> List[str]:
    q = ' '.join((question or '').split()).strip()
    if not q:
        return []

    results: List[str] = []

    try:
        tokenizer, model = _load_generation_model()

        prompts = [
            f'اعادة صياغة السؤال التالي بعدة طرق مختلفة مع الحفاظ على نفس المعنى: {q}',
            f'اكتب خمس صيغ مختلفة لنفس السؤال: {q}',
        ]

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
            outputs = model.generate(
                **inputs,
                max_length=96,
                num_return_sequences=4,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )
            for out in outputs:
                text = tokenizer.decode(out, skip_special_tokens=True)
                text = _clean_generated_text(text)
                if not text:
                    continue
                if text.lower() == q.lower():
                    continue
                results.append(text)
    except Exception as e:
        print('⚠️ paraphrase generation fallback:', e)

    clean = []
    seen = set()
    for item in results:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(item)

    if not clean:
        return _fallback_paraphrases(q)

    return clean[:5]


def load_cluster_variants() -> Dict[int, List[str]]:
    if not os.path.exists(CLUSTERS_JSON):
        return {}

    try:
        with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mapping: Dict[int, List[str]] = {}
        for item in data:
            faq_id = item.get('faq_id')
            variants = item.get('generated_variants') or []
            if faq_id is None:
                continue

            clean = []
            seen = set()
            for v in variants:
                v = ' '.join((v or '').split()).strip()
                if not v:
                    continue
                key = v.lower()
                if key in seen:
                    continue
                seen.add(key)
                clean.append(v)

            mapping[int(faq_id)] = clean

        return mapping
    except Exception as e:
        print('⚠️ cluster json read error:', e)
        return {}


def merge_paraphrases(question: str, current: str, cluster_variants: List[str], generated: List[str], max_items: int = 8) -> str:
    final = []
    seen = set()

    current_items = [p.strip() for p in (current or '').split('||') if p.strip()]

    for item in current_items + cluster_variants + generated:
        item = ' '.join((item or '').split()).strip()
        if not item:
            continue
        key = item.lower()
        if key == question.lower() or key in seen:
            continue
        seen.add(key)
        final.append(item)
        if len(final) >= max_items:
            break

    return ' || '.join(final)


@admin_bp.route("/admin")
def admin_dashboard():
    section = (request.args.get("section") or "").strip()
    faq_q = (request.args.get("faq_q") or "").strip()
    unanswered_q = (request.args.get("unanswered_q") or "").strip()

    conn = get_conn()
    c = conn.cursor()

    # إجمالي الأسئلة
    c.execute("SELECT COUNT(*) AS cnt FROM faqs")
    total_faqs = c.fetchone()["cnt"]

    # إجمالي غير المُجاب / منخفض الثقة
    c.execute("""
        SELECT COUNT(*) AS cnt
        FROM conversations
        WHERE matched_faq_id IS NULL
           OR confidence IS NULL
           OR confidence < ?
    """, (LOW_CONFIDENCE_THRESHOLD,))
    total_unanswered = c.fetchone()["cnt"]

    # عدد الأسئلة التي تحتوي paraphrases
    c.execute("""
        SELECT COUNT(*) AS cnt
        FROM faqs
        WHERE paraphrase IS NOT NULL
          AND TRIM(paraphrase) <> ''
    """)
    total_with_paraphrases = c.fetchone()["cnt"]

    # عدد الأسئلة التي لا تحتوي paraphrases
    c.execute("""
        SELECT COUNT(*) AS cnt
        FROM faqs
        WHERE paraphrase IS NULL
           OR TRIM(paraphrase) = ''
    """)
    total_without_paraphrases = c.fetchone()["cnt"]

    # الأسئلة مع فلترة
    faq_sql = """
        SELECT
            id,
            question,
            answer,
            COALESCE(category, '') AS category,
            COALESCE(paraphrase, '') AS paraphrase,
            COALESCE(created_at, '') AS created_at
        FROM faqs
    """
    faq_params = []

    if faq_q:
        faq_sql += """
            WHERE question LIKE ?
               OR answer LIKE ?
               OR category LIKE ?
               OR paraphrase LIKE ?
        """
        like_val = f"%{faq_q}%"
        faq_params.extend([like_val, like_val, like_val, like_val])

    faq_sql += " ORDER BY id DESC"
    c.execute(faq_sql, faq_params)
    faqs = c.fetchall()

    # غير المُجاب مع فلترة
    unanswered_sql = """
        SELECT
            rowid,
            COALESCE(timestamp, '') AS timestamp,
            COALESCE(user_message, '') AS user_message,
            confidence,
            matched_faq_id
        FROM conversations
        WHERE matched_faq_id IS NULL
           OR confidence IS NULL
           OR confidence < ?
    """
    unanswered_params = [LOW_CONFIDENCE_THRESHOLD]

    if unanswered_q:
        unanswered_sql += " AND user_message LIKE ?"
        unanswered_params.append(f"%{unanswered_q}%")

    unanswered_sql += " ORDER BY timestamp DESC LIMIT 200"
    c.execute(unanswered_sql, unanswered_params)
    unanswered = c.fetchall()

    conn.close()

    return render_template(
        "admin.html",
        faqs=faqs,
        unanswered=unanswered,
        total_faqs=total_faqs,
        total_unanswered=total_unanswered,
        total_with_paraphrases=total_with_paraphrases,
        total_without_paraphrases=total_without_paraphrases,
        faq_q=faq_q,
        unanswered_q=unanswered_q,
        section=section,
    )


@admin_bp.route("/admin/add", methods=["POST"])
def add_faq():
    question = (request.form.get("question") or "").strip()
    answer = (request.form.get("answer") or "").strip()
    category = (request.form.get("category") or "general").strip() or "general"
    paraphrase = (request.form.get("paraphrase") or "").strip()

    if not question or not answer:
        return redirect(url_for("admin.admin_dashboard", section="add-section"))

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO faqs (question, answer, category, paraphrase, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        category,
        paraphrase,
        datetime.utcnow().isoformat(),
    ))

    # إذا نفس السؤال كان ضمن غير المُجاب، نحذفه من القائمة
    c.execute("""
        DELETE FROM conversations
        WHERE user_message = ?
          AND (
                matched_faq_id IS NULL
                OR confidence IS NULL
                OR confidence < ?
          )
    """, (question, LOW_CONFIDENCE_THRESHOLD))

    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard", section="faq-section"))


@admin_bp.route("/admin/edit/<int:faq_id>", methods=["POST"])
def edit_faq(faq_id):
    question = (request.form.get("question") or "").strip()
    answer = (request.form.get("answer") or "").strip()
    category = (request.form.get("category") or "general").strip() or "general"
    paraphrase = (request.form.get("paraphrase") or "").strip()

    if not question or not answer:
        return redirect(url_for("admin.admin_dashboard", section="faq-section"))

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        UPDATE faqs
        SET question = ?, answer = ?, category = ?, paraphrase = ?
        WHERE id = ?
    """, (
        question,
        answer,
        category,
        paraphrase,
        faq_id,
    ))

    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard", section="faq-section"))


@admin_bp.route("/admin/delete/<int:faq_id>", methods=["POST"])
def delete_faq(faq_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM faqs WHERE id = ?", (faq_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard", section="faq-section"))


@admin_bp.route("/admin/unanswered/clear", methods=["POST"])
def clear_unanswered():
    """
    مسح الأسئلة غير المجابة أو منخفضة الثقة من جدول conversations.
    """
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        DELETE FROM conversations
        WHERE matched_faq_id IS NULL
           OR confidence IS NULL
           OR confidence < ?
    """, (LOW_CONFIDENCE_THRESHOLD,))

    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard", section="faq-section"))


@admin_bp.route("/admin/export/faqs")
def export_faqs():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT
            id,
            question,
            answer,
            COALESCE(category, '') AS category,
            COALESCE(paraphrase, '') AS paraphrase,
            COALESCE(created_at, '') AS created_at
        FROM faqs
        ORDER BY id
    """)
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "question", "answer", "category", "paraphrase", "created_at"])

    for row in rows:
        writer.writerow([
            row["id"],
            row["question"],
            row["answer"],
            row["category"],
            row["paraphrase"],
            row["created_at"],
        ])

    csv_data = "\ufeff" + output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=faqs_export.csv"
        },
    )


@admin_bp.route("/admin/import/faqs", methods=["POST"])
def import_faqs():
    file = request.files.get("file")
    if not file or not file.filename:
        return redirect(url_for("admin.admin_dashboard", section="faq-section"))

    filename = file.filename.lower()
    inserted = 0

    conn = get_conn()
    c = conn.cursor()

    try:
        if filename.endswith(".csv"):
            content = file.stream.read().decode("utf-8-sig", errors="ignore")
            reader = csv.DictReader(io.StringIO(content))

            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                category = (row.get("category") or "general").strip() or "general"
                paraphrase = (row.get("paraphrase") or "").strip()

                if not question or not answer:
                    continue

                c.execute("""
                    INSERT INTO faqs (question, answer, category, paraphrase, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    question,
                    answer,
                    category,
                    paraphrase,
                    datetime.utcnow().isoformat(),
                ))
                inserted += 1

        elif filename.endswith(".json"):
            content = file.stream.read().decode("utf-8", errors="ignore")
            data = json.loads(content)

            for row in data:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                category = (row.get("category") or "general").strip() or "general"
                paraphrase = (row.get("paraphrase") or "").strip()

                if not question or not answer:
                    continue

                c.execute("""
                    INSERT INTO faqs (question, answer, category, paraphrase, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    question,
                    answer,
                    category,
                    paraphrase,
                    datetime.utcnow().isoformat(),
                ))
                inserted += 1

        conn.commit()

    except Exception as e:
        print("⚠️ import error:", e)

    finally:
        conn.close()

    return redirect(url_for("admin.admin_dashboard", section="faq-section"))


# =========================
# System actions (new)
# =========================
@admin_bp.route("/admin/system/generate-paraphrases", methods=["POST"])
def system_generate_paraphrases():
    conn = get_conn()
    c = conn.cursor()

    cluster_map = load_cluster_variants()

    c.execute("""
        SELECT id, question, COALESCE(paraphrase, '') AS paraphrase
        FROM faqs
        ORDER BY id DESC
    """)
    rows = c.fetchall()

    updated = 0
    for row in rows:
        faq_id = int(row['id'])
        question = ' '.join((row['question'] or '').split()).strip()
        current = (row['paraphrase'] or '').strip()

        if not question:
            continue

        cluster_variants = cluster_map.get(faq_id, [])
        generated = []

        # استخدم clustering أولاً، وإذا ماكو variants كاملة نكمّل بالتوليد
        current_count = len([p for p in current.split('||') if p.strip()])
        if len(cluster_variants) == 0 and current_count == 0:
            generated = generate_paraphrases_text(question)

        final_text = merge_paraphrases(
            question=question,
            current=current,
            cluster_variants=cluster_variants,
            generated=generated,
            max_items=8,
        )

        if final_text != current:
            c.execute(
                "UPDATE faqs SET paraphrase = ? WHERE id = ?",
                (final_text, faq_id)
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f'✅ paraphrases generated/updated for {updated} FAQs')
    return redirect(url_for('admin.admin_dashboard', section='system-section'))


@admin_bp.route("/admin/system/rebuild-index", methods=["POST"])
def system_rebuild_index():
    try:
        from rag_index import build_index
        build_index(reset=True)
        print('✅ Index rebuilt from admin panel.')
    except Exception as e:
        print('⚠️ rebuild index error:', e)
    return redirect(url_for('admin.admin_dashboard', section='system-section'))
