import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import build_dataset, load_dataset
from .models import make_model, parse_interval
from .prompts import build_messages

ROOT = Path(__file__).resolve().parent.parent


def make_jobs(questions, model_spec, offsets, styles, pct):
    jobs = []
    for q in questions:
        jobs.append(
            dict(question_id=q["id"], model=model_spec, pct=pct, condition="baseline",
                 hint_style=None, offset=None, hint=None, truth=q["answer"], question=q["question"])
        )
        for style in styles:
            for off in offsets:
                jobs.append(
                    dict(question_id=q["id"], model=model_spec, pct=pct, condition="hint",
                         hint_style=style, offset=off, hint=q["answer"] + off,
                         truth=q["answer"], question=q["question"])
                )
    return jobs


def job_key(j):
    return (j["question_id"], j["model"], j["pct"], j["condition"], j["hint_style"], j["offset"])


def run_job(model, job):
    system, user = build_messages(job["question"], job["pct"], job["hint"], job["hint_style"])
    text = model.complete(system, user, meta=job)
    interval = parse_interval(text)
    row = {k: v for k, v in job.items() if k != "question"}
    row["raw"] = text
    row["lower"], row["upper"] = (interval if interval else (None, None))
    row["parsed"] = interval is not None
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mock"],
                    help="e.g. anthropic:claude-haiku-4-5 anthropic:claude-sonnet-5-5:low mock")
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--offsets", nargs="+", type=int, default=[-5, -2, 0, 2, 5],
                    help="hint = truth + offset, in base-10 exponent units")
    ap.add_argument("--styles", nargs="+", default=["soft"],
                    choices=["soft", "confident", "authority"])
    ap.add_argument("--pct", type=int, default=95, help="nominal confidence level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-name", default="main")
    args = ap.parse_args(argv)

    data_path = ROOT / "data" / f"questions_n{args.n_questions}_s{args.seed}.json"
    cache_path = ROOT / "data" / "raw_fermi.json"
    if data_path.exists():
        questions = load_dataset(data_path)
    else:
        questions = build_dataset(args.n_questions, args.seed, data_path, cache_path)
    print(f"{len(questions)} questions", file=sys.stderr)

    out_path = ROOT / "results" / f"{args.run_name}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["question_id"], r["model"], r["pct"], r["condition"], r["hint_style"], r["offset"]))

    for spec in args.models:
        model = make_model(spec)
        jobs = [j for j in make_jobs(questions, spec, args.offsets, args.styles, args.pct)
                if job_key(j) not in done]
        print(f"{spec}: {len(jobs)} queries to run", file=sys.stderr)
        with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(run_job, model, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    row = fut.result()
                except Exception as e:
                    j = futures[fut]
                    print(f"FAILED {j['question_id']} {j['condition']} {j['offset']}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(jobs)}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
