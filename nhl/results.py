"""Vérifie les prédictions finales archivées contre les vrais résultats
et alimente l'écran « Fiabilité » (docs/precision.json).

    python -m nhl.results
"""
import json
from datetime import date, timedelta

from . import config as C
from . import sources as S

MIN_NIGHTS_LIVE = 10     # en dessous, l'écran affiche le backtest


def grade():
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded"):
            continue
        goals, done, dropped = {}, True, set()
        old = date.fromisoformat(d["date"]) < date.today() - timedelta(days=2)
        for g in d["games"]:
            res = S.boxscore_goals(g["id"])
            if res is None:
                if old:                      # match reporté / annulé : on l'écarte
                    dropped |= {g["away"]["abbrev"], g["home"]["abbrev"]}
                    continue
                done = False
                continue
            goals.update(res)
        if not done:
            continue
        for pid, c in d["players"].items():
            c["scored"] = None if c["team"] in dropped else goals.get(int(pid), 0)
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


def tickets(players):
    """Mêmes tickets que la page Tickets : les plus probables, un joueur par équipe."""
    out, teams = [], set()
    for c in sorted(players.values(), key=lambda c: -c["prob"]):
        if c["team"] in teams:
            continue
        out.append(c)
        teams.add(c["team"])
        if len(out) == 3:
            break
    res = {}
    for name, n in (("Solo", 1), ("Duo", 2), ("Trio", 3)):
        pick = out[:n]
        prob = 1.0
        for c in pick:
            prob *= c["prob"]
        res[name] = {"players": [{"name": c["name"], "team": c["team"], "prob": c["prob"],
                                  "scored": (c.get("scored") or 0) > 0} for c in pick],
                     "prob": round(prob, 4), "passed": all((c.get("scored") or 0) > 0 for c in pick)}
    return res


def recap():
    """Récap de la dernière soirée évaluée → docs/recap.json (onglet Bilan + message du matin)."""
    nights = []
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded") and d.get("players"):
            nights.append(d)
    if not nights:
        return
    d = nights[-1]
    pl = {k: c for k, c in d["players"].items() if c.get("scored") is not None}
    scorers = [c for c in pl.values() if c["scored"] > 0]
    top = [d["players"][str(i)] for i in d["top"] if str(i) in pl]
    hot = [c for c in pl.values() if c.get("mood") == "hot"]
    cold = [c for c in pl.values() if c.get("mood") == "cold"]

    def mini(c):
        return {"name": c["name"], "team": c["team"], "opp": c["opp"], "prob": c["prob"],
                "rank": c.get("rank"), "goals": c.get("scored") or 0, "mood": c.get("mood")}

    def hits(n):
        return sum(1 for c in n if (c.get("scored") or 0) > 0)

    games = [{"away": g["away"]["abbrev"], "home": g["home"]["abbrev"]} for g in d["games"]]
    out = {
        "date": d["date"], "stage": d.get("stage"), "games": games,
        "expected": round(sum(c["prob"] for c in pl.values()), 1), "actual": len(scorers),
        "top10": [mini(c) for c in top[:10]], "top10_hits": hits(top[:10]),
        "tickets": tickets(pl),
        "surprises": [mini(c) for c in sorted(scorers, key=lambda c: c["prob"])[:3]],
        "hot": {"n": len(hot), "scored": hits(hot), "expected": round(sum(c["prob"] for c in hot), 1)},
        "cold": {"n": len(cold), "scored": hits(cold), "expected": round(sum(c["prob"] for c in cold), 1)},
        "history": [{"date": n["date"],
                     "top10_hits": hits([n["players"][str(i)] for i in n["top"][:10] if str(i) in n["players"]]),
                     "solo": tickets({k: c for k, c in n["players"].items() if c.get("scored") is not None})["Solo"]["passed"]}
                    for n in nights[-14:]],
    }
    (C.DOCS / "recap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ recap.json ({out['date']} : {out['top10_hits']}/10 dans le top, {out['actual']} buteurs pour {out['expected']} attendus)")


if __name__ == "__main__":
    grade()
    summarize()
    recap()
