"""python -m resume_ranker {rank,train} ..."""
import argparse
import json
import sys
from pathlib import Path

from . import parse

ROOT = Path(__file__).resolve().parent.parent


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
    t = sub.add_parser("train", help="train an NER model from data/")
    t.add_argument("kind", choices=["resume", "jd"])
    t.add_argument("--gpu", action="store_true", help="transformer config; needs spacy-transformers + CUDA")
    t.set_defaults(fn=cmd_train)
    args = p.parse_args(argv)
    args.fn(args)
