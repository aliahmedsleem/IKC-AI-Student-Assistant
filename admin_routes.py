# admin_routes.py
# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime

DB_PATH = "faqs.db"
admin_bp = Blueprint("admin", __name__, template_folder="templates")


def get_conn():
    return sqlite3.connect(DB_PATH)


@admin_bp.route("/admin")
def admin_dashboard():
    conn = get_conn()
    c = conn.cursor()

    # FAQs
    c.execute("SELECT id, question, answer, COALESCE(category,'') FROM faqs ORDER BY id DESC")
    faqs = c.fetchall()

    # unanswered / low-confidence
    c.execute("""
        SELECT rowid, timestamp, user_message, confidence
        FROM conversations
        WHERE matched_faq_id IS NULL OR confidence IS NULL OR confidence > 0.45
        ORDER BY timestamp DESC
        LIMIT 200
    """)
    unanswered = c.fetchall()

    conn.close()
    return render_template("admin.html", faqs=faqs, unanswered=unanswered)


@admin_bp.route("/admin/add", methods=["POST"])
def add_faq():
    question = (request.form.get("question") or "").strip()
    answer = (request.form.get("answer") or "").strip()
    category = (request.form.get("category") or "general").strip() or "general"

    if not question or not answer:
        return redirect(url_for("admin.admin_dashboard"))

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO faqs (question, answer, category, created_at) VALUES (?,?,?,?)",
        (question, answer, category, datetime.utcnow().isoformat()),
    )

    # (اختياري) تنظيف unanswered اللي نفس نص السؤال بالضبط
    c.execute("""
        DELETE FROM conversations
        WHERE user_message = ?
          AND (matched_faq_id IS NULL OR confidence IS NULL OR confidence > 0.45)
    """, (question,))

    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.route("/admin/edit/<int:faq_id>", methods=["POST"])
def edit_faq(faq_id):
    question = (request.form.get("question") or "").strip()
    answer = (request.form.get("answer") or "").strip()
    category = (request.form.get("category") or "general").strip() or "general"

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE faqs SET question=?, answer=?, category=? WHERE id=?",
        (question, answer, category, faq_id),
    )
    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.route("/admin/delete/<int:faq_id>", methods=["POST"])
def delete_faq(faq_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM faqs WHERE id=?", (faq_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.route("/admin/unanswered/clear", methods=["POST"])
def clear_unanswered():
    """تصفير/مسح الأسئلة غير المجابة أو منخفضة الجودة من جدول conversations."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        DELETE FROM conversations
        WHERE matched_faq_id IS NULL OR confidence IS NULL OR confidence > 0.45
    """)
    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard"))