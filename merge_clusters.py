# merge_clusters.py
# -*- coding: utf-8 -*-

import sqlite3
import json

DB_PATH = "faqs.db"
INPUT_FILE = "filtered_clusters/accepted.json"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# إضافة عمود paraphrase إذا غير موجود
try:
    cur.execute("ALTER TABLE faqs ADD COLUMN paraphrase TEXT")
    print("➕ Added paraphrase column")
except Exception:
    print("✔ paraphrase column exists")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

updated = 0

for item in data:
    question = item["question"]
    variants = item.get("variants", [])

    if not variants:
        continue

    variants_text = " || ".join(variants)

    cur.execute("""
        UPDATE faqs
        SET paraphrase = ?
        WHERE question = ?
    """, (variants_text, question))

    updated += 1

conn.commit()
conn.close()

print(f"✅ Done! Updated {updated} questions.")