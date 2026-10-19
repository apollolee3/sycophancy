import json
import random
import re
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"

DEPENDENT_PATTERN = re.compile(
    r"previous (problem|question|answer)|above|question \d|#\d|your answer to",
    re.IGNORECASE,
)

MALFORMED_PATTERN = re.compile(r"\b(is|base)\s+[?.](\s|$)")

MIN_EXPONENT = -20
MAX_EXPONENT = 40


def fetch_raw(cache_path: Path) -> list[dict]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    text = urllib.request.urlopen(RAW_URL, timeout=30).read().decode("utf-8")
    text = text[text.index("[") : text.rindex("]") + 1]
    items = json.loads(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(items))
    return items


def clean(items: list[dict]) -> list[dict]:
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
    items = clean(fetch_raw(cache_path))
    rng = random.Random(seed)
    rng.shuffle(items)
    sample = items[:n]
    for i, it in enumerate(sample):
        it["id"] = f"q{i:04d}"
    sample = [it for it in sample if not MALFORMED_PATTERN.search(it["question"])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=1))
    return sample


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())
