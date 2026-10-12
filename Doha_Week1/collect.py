"""Local helpers used by the existing shared dataset preparation script.

Source discovery and network download commands are not part of this package.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
