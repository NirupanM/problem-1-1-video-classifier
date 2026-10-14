"""Video classifier CLI.
  python src/main.py demo                    # synthetic data: generate, train, evaluate
  python src/main.py train DATA_DIR          # DATA_DIR/<class>/*.mp4
  python src/main.py classify VIDEO [...]    # JSON lines: class, confidence, chunking strategy
"""
import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import extract  # noqa: E402
from model import Model, CLASSES  # noqa: E402

MODEL_PATH = Path(__file__).parent / "model.json"
EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def feats(paths, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(extract, map(str, paths))))


def cmd_demo(a):
    import synth
    tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    print("videos in", tmp)
    paths, y, split = [], [], []
    for ci, c in enumerate(CLASSES):
        for s in range(a.per_class):
            p = tmp / f"{c}_{s}.mp4"
            synth.write(c, p, seed=1000 * ci + s)
            paths.append(p)
            y.append(ci)
            split.append(s % 3 == 0)  # ~1/3 held out
    y, test = np.array(y), np.array(split)
    t0 = time.time()
    X = feats(paths, a.workers)
    dt = time.time() - t0
    m = Model().fit(X[~test], y[~test])
    pred = m.proba(X[test]).argmax(1)
    print(f"features: {dt / len(paths):.2f}s/video wall ({a.workers} workers), {X.shape[1]} dims")
    print(f"held-out synthetic accuracy: {(pred == y[test]).mean():.2%} ({test.sum()} videos)")
    conf = np.zeros((len(CLASSES),) * 2, int)
    for t, p in zip(y[test], pred):
        conf[t, p] += 1
    print("confusion (rows=true), class order:", CLASSES)
    print(conf)
    m.fit(X, y)
    m.save(MODEL_PATH)
    print("saved", MODEL_PATH)


def cmd_train(a):
    paths, y = [], []
    for ci, c in enumerate(CLASSES):
        for p in sorted((Path(a.data) / c).glob("*")):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                y.append(ci)
    if not paths:
        sys.exit(f"no videos under {a.data}/<class>/ for classes {CLASSES}")
    Model().fit(feats(paths, a.workers), np.array(y)).save(MODEL_PATH)
    print(f"trained on {len(y)} videos -> {MODEL_PATH}")


def cmd_classify(a):
    m = Model.load(MODEL_PATH)
    for p in a.videos:
        try:
            c, conf, strat = m.predict(extract(p), a.min_conf)
            print(json.dumps(dict(video=p, cls=c, confidence=round(conf, 3), chunking=strat)))
        except IOError as e:
            print(json.dumps(dict(video=p, error=str(e))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo")
    d.add_argument("--per-class", type=int, default=9)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--out", help="folder to keep the generated videos in (default: temp dir)")
    t =sp.add_parser("train")
    t.add_argument("data")
    t.add_argument("--workers", type=int, default=4)
    c = sp.add_parser("classify")
    c.add_argument("videos", nargs="+")
    c.add_argument("--min-conf", type=float, default=0.6)
    args = ap.parse_args()
    {"demo": cmd_demo, "train": cmd_train, "classify": cmd_classify}[args.cmd](args)
