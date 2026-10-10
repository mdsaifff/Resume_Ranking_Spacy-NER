"""python -m resume_ranker {rank,train} ..."""
import argparse
import csv
import json
import sys
from pathlib import Path

from . import parse

ROOT = Path(__file__).resolve().parent.parent


def cmd_rank(args):
    from .rank import rank, score_all, to_rows
    from .score import load_embedder

    jd_docs, warnings = parse.read_all([(args.jd.name, args.jd.read_bytes())])
    if not jd_docs:
        sys.exit(f"error: {warnings[0]}")
    cvs, more = parse.read_all((p.name, p.read_bytes()) for p in sorted(args.resumes.iterdir()) if p.is_file())
    for w in warnings + more:
        print(f"warning: {w}", file=sys.stderr)
    if not cvs:
        sys.exit("error: no readable resumes")

    jd = parse.parse_jd(parse.load_nlp("jd"), *jd_docs[0])
    if not jd.skills:
        print("warning: no skills found in JD; skills score uses overall fit", file=sys.stderr)
    resume_nlp = parse.load_nlp("resume")
    scored = score_all(jd, [parse.parse_resume(resume_nlp, *d) for d in cvs], load_embedder())
    rows = to_rows(rank(scored, must_have=args.must_have))

    for r in rows[: args.top]:
        flag = f"  DISQUALIFIED (missing {r['disqualified']})" if r["disqualified"] else ""
        print(f"{r['rank']:>3}. {r['score']:5.1f}  {r['name']}{flag}")
        print(f"       missing: {r['missing'] or '-'}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


def cmd_train(args):
    from spacy.cli.train import train

    data, out = ROOT / "data" / args.kind, parse.MODELS_DIR / args.kind
    train(
        ROOT / "configs" / ("gpu.cfg" if args.gpu else "cpu.cfg"), out,
        use_gpu=0 if args.gpu else -1,
        overrides={"paths.train": str(data / "train.spacy"), "paths.dev": str(data / "test.spacy")},
    )
    perf = json.loads((out / "model-best" / "meta.json").read_text(encoding="utf-8"))["performance"]
    print(f"{args.kind}: overall F1 {perf['ents_f']:.3f}")
    for label, s in sorted(perf["ents_per_type"].items()):
        print(f"  {label:<22} P {s['p']:.3f}  R {s['r']:.3f}  F1 {s['f']:.3f}")


def main(argv: list[str] | None = None):
    for stream in (sys.stdout, sys.stderr):  # resumes contain non-cp1252 characters on Windows
        getattr(stream, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="resume_ranker")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="rank resumes against a job description")
    r.add_argument("jd", type=Path, help="job description (.pdf or .txt)")
    r.add_argument("resumes", type=Path, help="folder of resumes (.pdf or .txt)")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--csv", type=Path, help="write the full ranking here")
    r.add_argument("--must-have", nargs="*", default=[], metavar="SKILL")
    r.set_defaults(fn=cmd_rank)
    t = sub.add_parser("train", help="train an NER model from data/")
    t.add_argument("kind", choices=["resume", "jd"])
    t.add_argument("--gpu", action="store_true", help="transformer config; needs spacy-transformers + CUDA")
    t.set_defaults(fn=cmd_train)
    args = p.parse_args(argv)
    args.fn(args)
