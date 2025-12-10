# admin_routes.py
# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import sqlite3
from datetime import datetime

from config_utils import get_alpha, set_alpha

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
    c.execute(
        """
        SELECT rowid, timestamp, user_message, confidence
        FROM conversations
        WHERE matched_faq_id IS NULL OR confidence < 0.4
        ORDER BY timestamp DESC
        LIMIT 150
        """
    )
    unanswered = c.fetchall()

    conn.close()

    alpha = get_alpha()
    mode = "High-Precision" if alpha >= 0.7 else "High-Recall"

    return render_template("admin.html", faqs=faqs, unanswered=unanswered, current_mode=mode)


@admin_bp.route("/admin/add", methods=["POST"])
def add_faq():
    question = request.form["question"].strip()
    answer = request.form["answer"].strip()
    category = request.form.get("category", "general").strip() or "general"

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO faqs (question, answer, category, created_at) VALUES (?,?,?,?)",
        (question, answer, category, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.route("/admin/edit/<int:faq_id>", methods=["POST"])
def edit_faq(faq_id):
    question = request.form["question"].strip()
    answer = request.form["answer"].strip()
    category = request.form.get("category", "general").strip() or "general"

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


@admin_bp.route("/admin/mode/toggle", methods=["POST"])
def toggle_mode():
    a = get_alpha()
    new_a = 0.5 if a >= 0.7 else 0.7
    set_alpha(new_a)
    return redirect(url_for("admin.admin_dashboard"))


@admin_bp.route("/admin/mode", methods=["GET"])
def get_mode():
    a = get_alpha()
    mode = "high" if a >= 0.7 else "normal"
    return jsonify({"alpha": a, "mode": mode})
