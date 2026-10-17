"""
Thin wrappers around model APIs, plus a mock model for dry runs.

Every model exposes `complete(system, user) -> str`. Parsing the JSON interval
out of the text happens in `parse_interval` so all models share one parser.
"""

import json
import os
import random
import re
import time

JSON_PATTERN = re.compile(r'\{\s*"L"\s*:\s*(-?\d+)\s*,\s*"U"\s*:\s*(-?\d+)\s*\}')


def parse_interval(text: str) -> tuple[int, int] | None:
    """Pull the last {"L": x, "U": y} out of a response. None if missing."""
    matches = JSON_PATTERN.findall(text)
    if not matches:
        return None
    lower, upper = int(matches[-1][0]), int(matches[-1][1])
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _retry(fn, attempts: int = 5):
    """Call fn with exponential backoff. API rate limits are the usual reason."""
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


class AnthropicModel:
    def __init__(self, model_id: str, max_tokens: int = 400):
        import anthropic  # lazy so the mock path has no dependency

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        def call():
            resp = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(block.text for block in resp.content if hasattr(block, "text"))

        return _retry(call)


class OpenAIModel:
    def __init__(self, model_id: str, max_tokens: int = 400):
        import openai

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        return _retry(call)


class MockModel:
    """
    A fake estimator for testing the pipeline without API calls.

    Behavior: its private belief is truth + noise. Without a hint it reports a
    too-narrow interval around that belief (overconfident, like the paper found).
    With a hint it drags its midpoint `sycophancy` of the way toward the hint.
    Setting sycophancy=0.5 means the dry-run plots look like a real effect.
    """

    def __init__(self, sycophancy: float = 0.5):
        self.sycophancy = sycophancy

    def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        meta = meta or {}
        truth = meta["truth"]
        # seed per question so baseline and hinted runs share the same private belief
        local_rng = random.Random(meta["question_id"])
        belief = truth + local_rng.gauss(0, 1.5)
        if meta.get("hint") is not None:
            belief = belief + self.sycophancy * (meta["hint"] - belief)
        half_width = 1  # deliberately narrow, like real models
        lower, upper = round(belief - half_width), round(belief + half_width)
        return f'Rough reasoning here. {{"L": {lower}, "U": {upper}}}'


def make_model(spec: str):
    """
    spec looks like 'anthropic:claude-haiku-5-5', 'openai:gpt-4o-mini', or 'mock'.
    """
    if spec == "mock":
        return MockModel()
    provider, _, model_id = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(model_id)
    if provider == "openai":
        return OpenAIModel(model_id)
    raise ValueError(f"Unknown model spec: {spec}")
