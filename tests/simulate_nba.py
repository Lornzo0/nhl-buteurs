"""Test de bout en bout du volet NBA sur des données SIMULÉES (sans réseau).

Chaque joueur a un « vrai » niveau caché (minutes, tirs, adresse) : si la
chaîne fonctionne, le top doit être occupé par les meilleurs marqueurs et
les probabilités doivent être calibrées.

    python -m tests.simulate_nba
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nba import config as C
from nba import sources as S

rng = np.random.default_rng(5)
TEAMS = ["BOS", "NYK", "LAL", "GSW", "DEN", "MIA", "OKC", "PHX"]
FIRST = ["Jalen", "Tyrese", "Luka", "Jayson", "Devin", "Anthony", "Paolo", "Scottie", "Cade", "Trae"]
LAST = ["Brown", "Green", "Young", "Mitchell", "Booker", "Davis", "Ball", "White", "Barnes", "Allen"]


def make_world():
    players, pid = {}, 1620000
    for t in TEAMS:
        for k in range(13):
            pid += 1
            mins = max(8, 36 - k * 2.3 + rng.normal(0, 2))
            players[pid] = {"team": t, "name": f"{random.choice(FIRST)} {random.choice(LAST)} {pid % 100}",
                            "min": mins, "usage": rng.gamma(6, 0.06),        # tirs / minute
                            "ts": rng.normal(0.57, 0.04), "starter": k < 5}
    return players


def simulate_season(season, players, rounds):
    day, sched = date(season, 10, 22), []
    for _ in range(rounds):
        for i, a in enumerate(TEAMS):
            for b in TEAMS[i + 1:]:
                day += timedelta(days=random.choice([0, 0, 1]))
                sched.append((day, *((a, b) if random.random() < 0.5 else (b, a))))
    rows, n = [], 0
    for d, home, away in sched:
        n += 1
        gid = f"002{str(season)[2:]}{n:05d}"
        pace = rng.normal(1.0, 0.05)
        for t, opp in ((home, away), (away, home)):
            for p, x in players.items():
                if x["team"] != t or random.random() < 0.06:
                    continue
                mins = max(2, rng.normal(x["min"], 4))
                fga = rng.poisson(x["usage"] * mins * pace)
                fta = rng.poisson(x["usage"] * mins * 0.3)
                fg3a = rng.binomial(fga, 0.4)
                pts_per_shot = 2 * x["ts"]
                pts = int(max(0, rng.normal((fga + 0.44 * fta) * pts_per_shot, 3)))
                rows.append({"player_id": p, "name": x["name"], "team": t, "opp": opp, "home": int(t == home),
                             "game_id": gid, "date": d.isoformat(), "season": season, "min": round(mins, 1),
                             "pts": pts, "fga": fga, "fgm": int(fga * 0.47), "fg3a": fg3a,
                             "fg3m": int(fg3a * 0.36), "fta": fta, "ftm": int(fta * 0.78),
                             "oreb": rng.poisson(1), "reb": rng.poisson(4), "ast": rng.poisson(3),
                             "tov": rng.poisson(1.5), "starter": float(x["starter"])})
    return pd.DataFrame(rows), sched


def main():
    random.seed(5)
    C.TRAIN_SEASONS, C.CURRENT_SEASON = [2024, 2025], 2026
    out = Path("/tmp/nba-sim")
    C.MODELS, C.HISTORY, C.DOCS, C.CACHE = out / "models", out / "history", out / "docs", out / "cache"
    for p in (C.MODELS, C.HISTORY, C.DOCS):
        p.mkdir(parents=True, exist_ok=True)
    for f in C.HISTORY.glob("*.json"):
        f.unlink()
    players = make_world()
    logs, last = {}, None
    for s in (2024, 2025, 2026):
        logs[s], sched = simulate_season(s, players, 6 if s < 2026 else 2)
        last = sched
    S.game_logs = lambda season: S._typed(logs[season]) if season in logs else pd.DataFrame(columns=S.LOG_COLS)
    night = last[-1][0] + timedelta(days=1)
    games = [{"game_id": f"00226{90001 + i}", "day": night.isoformat(), "start": f"{night}T00:00:00Z",
              "state": 1, "home": h, "away": a, "home_name": h, "away_name": a, "venue": f"Salle {h}"}
             for i, (h, a) in enumerate([("BOS", "NYK"), ("LAL", "GSW"), ("DEN", "MIA"), ("OKC", "PHX")])]
    S.schedule = lambda day: [dict(g) for g in games]
    injured = next(x["name"] for x in players.values() if x["team"] == "BOS")
    S.injuries = lambda: {S.norm_name(injured): "Out"}
    S.team_points = lambda: {"BOS": 118.5, "NYK": 109.5}

    from nba import predict, results, train
    print("=== ENTRAÎNEMENT ===")
    train.main()
    print("=== PRÉDICTION ===")
    predict.run(night, "final")
    data = json.loads((C.DOCS / "data.json").read_text())
    assert injured not in [c["name"] for c in data["players"].values()], "le blessé « Out » devrait être retiré"
    top = [data["players"][str(i)] for i in data["top"][:10]]
    print("\nTop 10 (vrai niveau caché entre parenthèses) :")
    true = {p: x["usage"] * x["min"] * 1.22 * 2 * x["ts"] for p, x in players.items()}
    for c in top:
        print(f"  {c['rank']:>2}. {c['name']:<22} {c['team']} {c['prob']*100:5.1f} %  "
              f"~{c['pts']:.1f} pts  (vrai ≈ {true[c['id']]:.1f})")
    pred = {int(k): v["prob"] for k, v in data["players"].items()}
    corr = np.corrcoef([pred[p] for p in pred], [true[p] for p in pred])[0, 1]
    print(f"\nCorrélation prédiction / vrai niveau : {corr:.2f}")
    assert corr > 0.6, "le modèle devrait retrouver les meilleurs marqueurs"
    print("Facteurs du n°1 :", [(f["param"], f["mult"]) for f in top[0]["factors"]][:6], "…")
    g0 = data["games"][0]
    print("Match 1 :", g0["away"]["abbrev"], g0["away"]["pts"], "@", g0["home"]["abbrev"], g0["home"]["pts"],
          "victoire domicile", g0["home_win"])

    print("=== RÉSULTATS ===")
    S.boxscore_points = lambda gid: {int(k): int(rng.normal(v["pts"], v["sd"]))
                                     for k, v in data["players"].items() if data["games"] and
                                     any(gm["id"] == gid and v["team"] in (gm["home"]["abbrev"], gm["away"]["abbrev"])
                                         for gm in data["games"])}
    results.MIN_NIGHTS_LIVE = 1
    results.grade()
    ns = results.nights()
    results.summarize(ns)
    results.recap(ns)
    rc = json.loads((C.DOCS / "recap.json").read_text())
    print(f"Récap : {rc['top10_hits']}/10 · {rc['actual']} joueurs à 20+ pour {rc['expected']} attendus · "
          f"Solo {'gagné' if rc['tickets']['Solo']['passed'] else 'perdu'}")
    print("OK")


if __name__ == "__main__":
    main()
