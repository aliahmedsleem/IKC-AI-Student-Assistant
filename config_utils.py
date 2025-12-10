# config_utils.py
import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "alpha": 0.7  # وزن الـ Embeddings داخل البحث الهجين (0.0–1.0)
}


def _ensure_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)


def get_alpha() -> float:
    _ensure_config()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("alpha", DEFAULT_CONFIG["alpha"]))
    except Exception:
        return DEFAULT_CONFIG["alpha"]


def set_alpha(value: float):
    _ensure_config()
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"alpha": float(value)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Unable to save alpha:", e)
