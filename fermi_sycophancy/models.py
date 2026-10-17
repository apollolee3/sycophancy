import json
import os
import random
import re
import time

JSON_PATTERN = re.compile(r'\{\s*"L"\s*:\s*(-?\d+)\s*,\s*"U"\s*:\s*(-?\d+)\s*\}')


def parse_interval(text: str) -> tuple[int, int] | None:
    matches = JSON_PATTERN.findall(text)
    if not matches:
        return None
    lower, upper = int(matches[-1][0]), int(matches[-1][1])
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _retry(fn, attempts: int = 5):
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


class AnthropicModel:
    def __init__(self, model_id: str, max_tokens: int = 400):
        import anthropic

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
    def __init__(self, sycophancy: float = 0.5):
        self.sycophancy = sycophancy

    def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        meta = meta or {}
        truth = meta["truth"]
        local_rng = random.Random(meta["question_id"])
        belief = truth + local_rng.gauss(0, 1.5)
        if meta.get("hint") is not None:
            belief = belief + self.sycophancy * (meta["hint"] - belief)
        half_width = 1
        lower, upper = round(belief - half_width), round(belief + half_width)
        return f'Rough reasoning here. {{"L": {lower}, "U": {upper}}}'


def make_model(spec: str):
    if spec == "mock":
        return MockModel()
    provider, _, model_id = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(model_id)
    if provider == "openai":
        return OpenAIModel(model_id)
    raise ValueError(f"Unknown model spec: {spec}")
