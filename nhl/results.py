"""Vérifie les prédictions finales archivées contre les vrais résultats
et alimente l'écran « Fiabilité » (docs/precision.json).

    python -m nhl.results
"""
import json

from . import config as C
from . import sources as S

MIN_NIGHTS_LIVE = 10     # en dessous, l'écran affiche le backtest


def grade():
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded"):
            continue
        goals, done = {}, True
        for g in d["games"]:
            res = S.boxscore_goals(g["id"])
            if res is None:
                done = False
                continue
            goals.update(res)
        if not done:
            continue
        for pid, c in d["players"].items():
            c["scored"] = goals.get(int(pid))
        d["graded"] = True
        f.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
        print(f"  {f.stem} : résultats ajoutés")


def summarize():
    nights = []
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded"):
            nights.append(d)
    model = json.loads((C.DOCS / "model.json").read_text()) if (C.DOCS / "model.json").exists() else {}
    out = {"source": "backtest", "backtest": model.get("backtest")}
    if nights:
        last = nights[-1]
        out["last_night"] = {"date": last["date"], "top": [
            {"name": last["players"][str(pid)]["name"], "prob": last["players"][str(pid)]["prob"],
             "scored": (last["players"][str(pid)].get("scored") or 0) > 0}
            for pid in last["top"][:5] if str(pid) in last["players"]]}
    if len(nights) >= MIN_NIGHTS_LIVE:
        ps, ys, hits = [], [], []
        for d in nights:
            for c in d["players"].values():
                if c.get("scored") is not None:
                    ps.append(c["prob"])
                    ys.append(1.0 if c["scored"] > 0 else 0.0)
            hits.append(sum(1 for pid in d["top"][:10]
                            if (d["players"].get(str(pid), {}).get("scored") or 0) > 0))
        cal = []
        for lo, hi in [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 1.0)]:
            idx = [i for i, p in enumerate(ps) if lo <= p < hi]
            if len(idx) >= 20:
                cal.append({"bin": f"{round(lo*100)}–{round(hi*100)} %" if hi < 1 else f"{round(lo*100)} %+",
                            "pred": round(sum(ps[i] for i in idx) / len(idx), 3),
                            "obs": round(sum(ys[i] for i in idx) / len(idx), 3), "n": len(idx)})
        out.update({"source": "live", "nights": len(nights), "n": len(ps),
                    "top10_hits": round(sum(hits) / len(hits), 2), "calibration": cal})
    (C.DOCS / "precision.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ precision.json ({out['source']}, {len(nights)} soirées évaluées)")


if __name__ == "__main__":
    grade()
    summarize()
