"""Vérifie les prédictions NBA archivées, alimente le Bilan (docs/nba/precision.json
et docs/nba/recap.json).

    python -m nba.results
"""
import json
from datetime import date, timedelta

from . import config as C
from . import sources as S

MIN_NIGHTS_LIVE = 10
BINS = [(0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]


def hit(c):
    return (c.get("pts_real") or 0) >= C.THRESHOLD


def _fallback_points():
    """Si la feuille de match ne répond pas : points de la saison en cours (même source, en cache)."""
    try:
        logs = S.game_logs(C.CURRENT_SEASON)
    except SystemExit:
        return {}
    out = {}
    for gid, pid, pts in zip(logs["game_id"], logs["player_id"], logs["pts"]):
        out.setdefault(gid, {})[int(pid)] = int(pts)
    return out


def grade():
    fallback = None
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded"):
            continue
        pts, done, dropped = {}, True, set()
        old = date.fromisoformat(d["date"]) < date.today() - timedelta(days=2)
        for g in d["games"]:
            res = S.boxscore_points(g["id"])
            if res is None:
                if fallback is None:
                    fallback = _fallback_points()
                res = fallback.get(str(g["id"]))
            if res is None:
                if old:                      # match reporté : on l'écarte
                    dropped |= {g["away"]["abbrev"], g["home"]["abbrev"]}
                    continue
                done = False
                continue
            pts.update(res)
        if not done:
            continue
        for pid, c in d["players"].items():
            # Absent de la feuille de match = n'a pas joué : prédiction écartée (pas un échec).
            c["pts_real"] = None if c["team"] in dropped else pts.get(int(pid))
        d["graded"] = True
        f.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
        print(f"  {f.stem} : résultats ajoutés")


def nights():
    out = []
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("graded") and d.get("players"):
            out.append(d)
    return out


def summarize(ns):
    model = json.loads((C.DOCS / "model.json").read_text()) if (C.DOCS / "model.json").exists() else {}
    out = {"source": "backtest", "backtest": model.get("backtest")}
    if ns:
        last = ns[-1]
        out["last_night"] = {"date": last["date"], "top": [
            {"name": last["players"][str(pid)]["name"], "prob": last["players"][str(pid)]["prob"],
             "scored": hit(last["players"][str(pid)]), "pts": last["players"][str(pid)].get("pts_real")}
            for pid in last["top"][:5] if str(pid) in last["players"]]}
    if len(ns) >= MIN_NIGHTS_LIVE:
        ps, ys, hits = [], [], []
        for d in ns:
            played = {k: c for k, c in d["players"].items() if c.get("pts_real") is not None}
            for c in played.values():
                ps.append(c["prob"])
                ys.append(1.0 if hit(c) else 0.0)
            hits.append(sum(1 for c in top_played(d)[:10] if hit(c)))
        cal = []
        for lo, hi in BINS:
            idx = [i for i, p in enumerate(ps) if lo <= p < hi]
            if len(idx) >= 20:
                cal.append({"bin": f"{round(lo*100)}–{round(hi*100)} %" if hi < 1 else f"{round(lo*100)} %+",
                            "pred": round(sum(ps[i] for i in idx) / len(idx), 3),
                            "obs": round(sum(ys[i] for i in idx) / len(idx), 3), "n": len(idx)})
        out.update({"source": "live", "nights": len(ns), "n": len(ps),
                    "top10_hits": round(sum(hits) / len(hits), 2), "calibration": cal})
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "precision.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ nba/precision.json ({out['source']}, {len(ns)} soirées évaluées)")


def top_played(d):
    """Top du soir parmi les joueurs qui ont effectivement joué."""
    return [d["players"][str(i)] for i in d["top"]
            if str(i) in d["players"] and d["players"][str(i)].get("pts_real") is not None]


def tickets(players):
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
                                  "pts": c.get("pts_real"), "scored": hit(c)} for c in pick],
                     "prob": round(prob, 4), "passed": bool(pick) and all(hit(c) for c in pick)}
    return res


def recap(ns):
    if not ns:
        return
    d = ns[-1]
    pl = {k: c for k, c in d["players"].items() if c.get("pts_real") is not None}
    scorers = [c for c in pl.values() if hit(c)]
    top = top_played(d)
    hot = [c for c in pl.values() if c.get("mood") == "hot"]
    cold = [c for c in pl.values() if c.get("mood") == "cold"]

    def mini(c):
        return {"name": c["name"], "team": c["team"], "opp": c["opp"], "prob": c["prob"],
                "rank": c.get("rank"), "pts": c.get("pts_real"), "hit": hit(c), "mood": c.get("mood")}

    def hits(n):
        return sum(1 for c in n if hit(c))

    out = {
        "sport": "nba", "threshold": C.THRESHOLD, "date": d["date"], "stage": d.get("stage"),
        "games": [{"away": g["away"]["abbrev"], "home": g["home"]["abbrev"]} for g in d["games"]],
        "expected": round(sum(c["prob"] for c in pl.values()), 1), "actual": len(scorers),
        "top10": [mini(c) for c in top[:10]], "top10_hits": hits(top[:10]),
        "tickets": tickets(pl),
        "surprises": [mini(c) for c in sorted(scorers, key=lambda c: c["prob"])[:3]],
        "hot": {"n": len(hot), "scored": hits(hot), "expected": round(sum(c["prob"] for c in hot), 1)},
        "cold": {"n": len(cold), "scored": hits(cold), "expected": round(sum(c["prob"] for c in cold), 1)},
        "history": [{"date": n["date"], "top10_hits": hits(top_played(n)[:10]),
                     "solo": tickets({k: c for k, c in n["players"].items() if c.get("pts_real") is not None})["Solo"]["passed"]}
                    for n in ns[-14:]],
    }
    (C.DOCS / "recap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ nba/recap.json ({out['date']} : {out['top10_hits']}/10 dans le top, "
          f"{out['actual']} joueurs à 20+ pour {out['expected']} attendus)")


if __name__ == "__main__":
    grade()
    ns = nights()
    summarize(ns)
    recap(ns)
