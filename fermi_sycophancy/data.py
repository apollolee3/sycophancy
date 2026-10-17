"""
Fetch and filter the open-scioly-fermi question set.

Source: https://github.com/landy8697/open-scioly-fermi (MIT license).
This is the same corpus FermiEval (Epstein et al., 2025) was built from.
Each item is a question plus an integer answer, which is a base-10 exponent:
answer = 13 means the true value is about 10^13.
"""

import json
import random
import re
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"

# Questions that reference another question are useless on their own.
DEPENDENT_PATTERN = re.compile(
    r"previous (problem|question|answer)|above|question \d|#\d|your answer to",
    re.IGNORECASE,
)

# Keep the answer range sane. Exponents outside this are usually parser errors
# ("Compute 2023^7" style stunts) rather than real estimation questions.
MIN_EXPONENT = -20
MAX_EXPONENT = 40


def fetch_raw(cache_path: Path) -> list[dict]:
    """Download data.js once and cache it as JSON."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    text = urllib.request.urlopen(RAW_URL, timeout=30).read().decode("utf-8")
    # data.js is `data = [ ... ]`, so slice out the JSON array
    text = text[text.index("[") : text.rindex("]") + 1]
    items = json.loads(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(items))
    return items


def clean(items: list[dict]) -> list[dict]:
    """Drop dependent questions, out-of-range answers, and duplicates."""
    seen = set()
    out = []
    for it in items:
        q = it["question"].strip()
        a = it["answer"]
        if not isinstance(a, int) or not (MIN_EXPONENT <= a <= MAX_EXPONENT):
            continue
        if DEPENDENT_PATTERN.search(q):
            continue
        if len(q) < 20:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "answer": a, "source": it.get("source", "")})
    return out


def build_dataset(n: int, seed: int, out_path: Path, cache_path: Path) -> list[dict]:
    """Sample n cleaned questions and write them with stable ids."""
    items = clean(fetch_raw(cache_path))
    rng = random.Random(seed)
    rng.shuffle(items)
    sample = items[:n]
    for i, it in enumerate(sample):
        it["id"] = f"q{i:04d}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=1))
    return sample


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())
