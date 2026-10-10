"""Pooled evaluation: the original notebook score vs the hybrid (+ ablations) on hand-labeled held-out data.

  python eval/evaluate.py list           # show held-out JDs with their indices
  python eval/evaluate.py pool 0 5 12    # build labeling pools -> eval/pool/*.txt + eval/labels.json
  python eval/evaluate.py score          # after labeling every pair 0-3: NDCG@5 / P@5 table

ponytail: the held-out .spacy files double as the dev set used to pick model-best, so NER quality
on them is slightly optimistic. Split off a separate test set if that ever matters.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LABELS, POOL = ROOT / "eval" / "labels.json", ROOT / "eval" / "pool"
POOL_SIZE, TOP_FROM_EACH = 15, 7


def dcg(rels: list[int]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[int], k: int = 5) -> float:
    ideal = dcg(sorted(rels, reverse=True)[:k])
    return dcg(rels[:k]) / ideal if ideal else 0.0


def precision_at_k(rels: list[int], k: int = 5, relevant: int = 2) -> float:
    return sum(r >= relevant for r in rels[:k]) / k


def held_out(kind: str) -> list[str]:
    import spacy
    from spacy.tokens import DocBin

    nlp = spacy.blank("en")
    return [d.text for d in DocBin().from_disk(ROOT / "data" / kind / "test.spacy").get_docs(nlp.vocab)]


def _parsed(jd_ids, cv_ids=None):
    from resume_ranker import parse
    from resume_ranker.score import load_embedder

    jd_texts, cv_texts = held_out("jd"), held_out("resume")
    jd_nlp, cv_nlp = parse.load_nlp("jd"), parse.load_nlp("resume")
    jds = {f"jd:{i}": parse.parse_jd(jd_nlp, f"jd:{i}", jd_texts[i]) for i in jd_ids}
    cvs = {f"resume:{i}": parse.parse_resume(cv_nlp, f"resume:{i}", cv_texts[i])
           for i in (range(len(cv_texts)) if cv_ids is None else cv_ids)}
    return jds, cvs, load_embedder()


def cmd_list(args):
    for i, text in enumerate(held_out("jd")):
        print(f"{i:>3}  {' '.join(text.split())[:100]}")


def cmd_pool(args):
    from resume_ranker.rank import rank, score_all
    from resume_ranker.score import legacy_score

    jds, cvs, embed = _parsed(args.jd)
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    POOL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for jd_name, jd in jds.items():
        legacy = sorted(cvs, key=lambda n: -legacy_score(jd, cvs[n]))[:TOP_FROM_EACH]
        hybrid = [r.name for r in rank(score_all(jd, list(cvs.values()), embed))][:TOP_FROM_EACH]
        pool = list(dict.fromkeys(legacy + hybrid))
        rest = [n for n in cvs if n not in pool]
        rng.shuffle(rest)
        pool += rest[: POOL_SIZE - len(pool)]
        rng.shuffle(pool)  # blind the labeler to which system picked each resume
        (POOL / f"{jd_name.replace(':', '_')}.txt").write_text(jd.text, encoding="utf-8")
        entry = labels.setdefault(jd_name, {})
        for n in pool:
            entry.setdefault(n, None)
            (POOL / f"{n.replace(':', '_')}.txt").write_text(cvs[n].text, encoding="utf-8")
    LABELS.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Texts in {POOL}. Replace every null in {LABELS} with 0-3 "
          "(3 = would interview, 2 = plausible, 1 = weak, 0 = irrelevant), then run: score")


def cmd_score(args):
    from resume_ranker.rank import rank, score_all
    from resume_ranker.score import WEIGHTS, legacy_score

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    unlabeled = sum(v is None for pool in labels.values() for v in pool.values())
    if unlabeled:
        sys.exit(f"{unlabeled} pairs in {LABELS} are still null; label them 0-3 first.")
    jd_ids = [int(k.split(":")[1]) for k in labels]
    cv_ids = sorted({int(n.split(":")[1]) for pool in labels.values() for n in pool})
    jds, cvs, embed = _parsed(jd_ids, cv_ids)
    systems = {"legacy (notebook)": None, "hybrid": WEIGHTS,
               **{f"hybrid without {k}": {**WEIGHTS, k: 0.0} for k in WEIGHTS}}
    sums = {name: [0.0, 0.0] for name in systems}
    for jd_name, rels in labels.items():
        jd, pool = jds[jd_name], [cvs[n] for n in rels]
        scored = score_all(jd, pool, embed)
        for name, weights in systems.items():
            if weights is None:
                order = [cv.name for cv in sorted(pool, key=lambda cv: -legacy_score(jd, cv))]
            else:
                order = [r.name for r in rank(scored, weights)]
            ranked = [rels[n] for n in order]
            sums[name][0] += ndcg_at_k(ranked)
            sums[name][1] += precision_at_k(ranked)
    print(f"{len(labels)} JDs, {sum(map(len, labels.values()))} labeled pairs\n")
    print("| system | NDCG@5 | P@5 |\n|---|---|---|")
    for name, (n, p) in sums.items():
        print(f"| {name} | {n / len(labels):.3f} | {p / len(labels):.3f} |")


def main():
    p = argparse.ArgumentParser(prog="evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    pool = sub.add_parser("pool")
    pool.add_argument("jd", type=int, nargs="+", help="held-out JD indices from `list`")
    pool.set_defaults(fn=cmd_pool)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
