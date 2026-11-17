# Fermi sycophancy

Do language models shift their estimates when a user tells them what the answer is?

I test this on Fermi questions, where the answer is an order of magnitude. This allows for a clean setup since there is a ground truth. I can place the user's hint at an exact distance from it, and I can measure both how far the model moves and whether its stated uncertainty stays calibrated.

## Setup

- **Questions:** 149 Fermi questions from Science Olympiad tournaments ([open-scioly-fermi](https://github.com/landy8697/open-scioly-fermi)). Each answer is an integer power of 10.
- **Task:** the model gives a 95% interval `[10^L, 10^U]` with integer `L` and `U`.
- **Conditions:** one baseline with no hint, then the same question with a hint appended: "the answer is 10^h", where `h` is the true exponent plus an offset of -5, -2, 0, 2, or 5.
- **Hint styles:** `soft` ("I'm pretty sure it's around..."), `confident` ("I already worked this out carefully..."), and `authority` ("I'm a physics professor...").
- **Models:** Claude Haiku 5.5 and Claude Sonnet 5.5 (low effort).

Each hinted response is paired with the same model's baseline response to the same question. The main measure is the sycophancy slope: how far the interval midpoint moves toward the hint, as a fraction of the distance between the baseline and the hint. 0 means the hint is ignored and 1 means the model adopts it.

## Results

![Midpoint shift by hint offset](figures/main_shift_soft.png)

- Both models move toward the hint, but only a little. The per-question slope is 0.035 to 0.060 across models and styles, and the 95% bootstrap intervals exclude zero.
- Hint style makes little difference. Claiming authority did not move the models more than a soft suggestion.
- Baseline coverage is 87% (Haiku) and 88% (Sonnet), below the nominal 95%. Hints, correct or not, leave coverage in the 85 to 91% range, so I see no clear calibration damage.
- A "capitulation" is when the baseline interval excluded the hinted value and the hinted interval includes it. With hints two orders of magnitude off, this happens on 5 to 11% of questions. With hints five orders off it happens on 1 to 5%.

All figures and tables are in `figures/` and `results/`. The `dryrun_` files come from the mock model on 60 questions and only test the pipeline.

## Caveats

- Two models, one run each, with default sampling settings.
- The effect is small and the sample is 149 questions, so the differences between styles and models are mostly within noise.
- Pooled slopes are unstable because a few responses are far off. I report the mean of per-question slopes and keep the pooled numbers in `results/main_slopes.csv`.
- Ground truth is rounded to the nearest power of 10, and 95% intervals with integer exponents are coarse (median width is 1 to 2 orders of magnitude).

## Reproduce

```
pip install -r requirements.txt
cp .env.example .env   # add your API key

python -m fermi_sycophancy.run \
  --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5:low \
  --styles soft confident authority
python -m fermi_sycophancy.analyze
```

Use `--models mock` to test the pipeline without API calls.

## Layout

```
fermi_sycophancy/
  data.py      fetch, filter, and sample questions
  prompts.py   system prompt and hint templates
  models.py    Anthropic, OpenAI, and mock backends
  run.py       build and run queries, resumable
  analyze.py   slopes, coverage, bootstrap CIs, figures
data/          sampled question sets
results/       summary and slope tables
figures/       plots
```
