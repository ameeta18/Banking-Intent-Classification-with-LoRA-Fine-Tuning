"""Shared paths and text normalization for BANKING77 scripts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()
