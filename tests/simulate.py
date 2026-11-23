"""Test de bout en bout sur des données SIMULÉES au format MoneyPuck / API NHL.

Sert à vérifier que toute la chaîne tourne (chargement → paramètres →
entraînement → prédiction → résultats → JSON de l'app) sans accès réseau.
Les joueurs ont un « vrai » talent caché : si le modèle fonctionne, il doit
classer les meilleurs tireurs en haut et être bien calibré.

    python -m tests.simulate
"""
import io
import json
import random
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhl import config as C
from nhl import sources as S

rng = np.random.default_rng(3)
TEAMS = ["TOR", "MTL", "BOS", "EDM", "COL", "TBL", "NYR", "VGK"]
FIRST = ["Alex", "Sam", "Max", "Leo", "Noah", "Eli", "Jack", "Owen", "Liam", "Theo", "Hugo", "Luca"]
LAST = ["Martin", "Roy", "Smith", "Larsen", "Novak", "Kane", "Berg", "Fox", "Stone", "Price", "Lind", "Ward"]


def make_world():
    players, goalies, pid = {}, {}, 8470000
    for t in TEAMS:
        for k in range(20):
            pid += 1
            d = k >= 14
            players[pid] = {"team": t, "pos": "D" if d else random.choice("CLR"),
                            "name": f"{random.choice(FIRST)} {random.choice(LAST)}",
                            "shoot": rng.gamma(4, 0.5) * (0.5 if d else 1.0),    # tirs / 20 min
                            "qual": rng.normal(0.075 if not d else 0.04, 0.012),
                            "finish": rng.normal(1.0, 0.08), "toi": rng.uniform(12, 21) + (2 if d else 0),
                            "pp": rng.random() < 0.3}
        for k in range(2):
            pid += 1
            goalies[pid] = {"team": t, "name": f"G{k} {random.choice(LAST)}", "skill": rng.normal(0, 0.06)}
    return players, goalies


def simulate_season(season, players, goalies, n_rounds=8):
    start = date(season, 10, 8)
    sched, day = [], start
    for _ in range(n_rounds):
        for i, a in enumerate(TEAMS):
            for b in TEAMS[i + 1:]:
                day += timedelta(days=random.choice([0, 0, 1]))
                home, away = (a, b) if random.random() < 0.5 else (b, a)
                sched.append((day, home, away))
    sk, gl, tm, gid = [], [], [], season * 1000000
    for d, home, away in sched:
        gid += 1
        starters = {t: random.choice([g for g, x in goalies.items() if x["team"] == t]) for t in (home, away)}
        team_xg = {}
        for t, opp in ((home, away), (away, home)):
            ha = "HOME" if t == home else "AWAY"
            gk = goalies[starters[opp]]
            txg, tg = 0.0, 0
            for p, x in players.items():
                if x["team"] != t or random.random() < 0.04:       # 4 % d'absents
                    continue
                toi = max(4, rng.normal(x["toi"], 2)) * 60
                fen = rng.poisson(x["shoot"] * toi / 1200 * (1.05 if ha == "HOME" else 1))
                ixg = sum(max(0.005, rng.normal(x["qual"], 0.04)) for _ in range(fen))
                goals = rng.poisson(max(0, ixg * x["finish"] * (1 - gk["skill"] * 3)))
                pp_t = (rng.uniform(60, 200) if x["pp"] else rng.uniform(0, 20))
                txg += ixg
                tg += goals
                base = dict(playerId=p, season=season, name=x["name"], gameId=gid, playerTeam=t,
                            opposingTeam=opp, home_or_away=ha, gameDate=int(d.strftime("%Y%m%d")),
                            position=x["pos"])
                hd = int(fen * 0.35)
                sk.append({**base, "situation": "all", "icetime": toi, "I_F_goals": goals, "I_F_xGoals": ixg,
                           "I_F_shotsOnGoal": int(fen * 0.7), "I_F_unblockedShotAttempts": fen,
                           "I_F_highDangerShots": hd, "I_F_mediumDangerShots": int(fen * 0.35),
                           "I_F_lowDangerShots": fen - hd - int(fen * 0.35), "OnIce_F_xGoals": ixg * 2.2})
                sk.append({**base, "situation": "5on4", "icetime": pp_t, "I_F_goals": 0,
                           "I_F_xGoals": ixg * pp_t / toi * 2, "I_F_shotsOnGoal": 0,
                           "I_F_unblockedShotAttempts": 0, "I_F_highDangerShots": 0,
                           "I_F_mediumDangerShots": 0, "I_F_lowDangerShots": 0, "OnIce_F_xGoals": 0})
            team_xg[t] = (txg, tg)
            gl.append(dict(playerId=starters[opp], season=season, name=gk["name"], gameId=gid,
                           playerTeam=opp, opposingTeam=t, gameDate=int(d.strftime("%Y%m%d")),
                           situation="all", icetime=3600, xGoals=txg, goals=tg))
        for t, opp in ((home, away), (away, home)):
            tm.append(dict(team=t, season=season, gameId=gid, playerTeam=t, opposingTeam=opp,
                           home_or_away="HOME" if t == home else "AWAY", gameDate=int(d.strftime("%Y%m%d")),
                           situation="all", iceTime=3600, xGoalsFor=team_xg[t][0], xGoalsAgainst=team_xg[opp][0],
                           goalsFor=team_xg[t][1], goalsAgainst=team_xg[opp][1],
                           penaltiesFor=rng.poisson(3.5), blockedShotAttemptsAgainst=rng.poisson(14)))
    return pd.DataFrame(sk), pd.DataFrame(gl), pd.DataFrame(tm), sched


def zipped(df):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("data.csv", df.to_csv(index=False))
    return b.getvalue()


def main():
    random.seed(3)
    C.TRAIN_SEASONS, C.CURRENT_SEASON = [2024, 2025], 2026
    from pathlib import Path
    out = Path("/tmp/nhl-sim")                      # rien n'est écrit dans le vrai projet
    C.MODELS, C.HISTORY, C.DOCS = out / "models", out / "history", out / "docs"
    for p in (C.MODELS, C.HISTORY, C.DOCS):
        p.mkdir(parents=True, exist_ok=True)
    players, goalies = make_world()
    blobs, teams_all, last_sched = {}, [], None
    for s in (2024, 2025, 2026):
        sk, gl, tm, sched = simulate_season(s, players, goalies, n_rounds=8 if s < 2026 else 3)
        blobs[f"mp_skaters_{s}.zip"], blobs[f"mp_goalies_{s}.zip"] = zipped(sk), zipped(gl)
        teams_all.append(tm)
        last_sched = sched
    blobs["mp_all_teams.csv"] = pd.concat(teams_all).to_csv(index=False).encode()

    # On branche les fausses sources à la place d'Internet.
    S._cached = lambda name, url, age: blobs.get(name)
    night = last_sched[-1][0] + timedelta(days=1)
    games = [{"game_id": 99000001 + i, "start": f"{night}T23:00:00Z", "state": "FUT",
              "home": h, "away": a, "home_name": h, "away_name": a, "venue": f"Aréna {h}"}
             for i, (h, a) in enumerate([("TOR", "MTL"), ("EDM", "COL"), ("BOS", "TBL"), ("NYR", "VGK")])]
    S.schedule = lambda day: [dict(g) for g in games]
    S.roster = lambda team: {p: {"name": x["name"], "pos": x["pos"]} for p, x in players.items() if x["team"] == team}
    S.goalie_starts = lambda team, n=10: [{"date": "2026-01-01", "id": g, "name": x["name"]}
                                          for g, x in goalies.items() if x["team"] == team][:1] * 6
    S.team_totals = lambda: {"TOR": 3.6, "MTL": 2.7}

    from nhl import train, predict, results
    print("=== ENTRAÎNEMENT ===")
    train.main()
    print("=== PRÉDICTION ===")
    predict.run(night, "final")

    data = json.loads((C.DOCS / "data.json").read_text())
    top = [data["players"][str(i)] for i in data["top"][:10]]
    print("\nTop 10 prédit (talent caché réel entre parenthèses) :")
    for c in top:
        x = players[c["id"]]
        print(f"  {c['rank']:>2}. {c['name']:<14} {c['team']} {c['prob']*100:5.1f} %   "
              f"(tirs {x['shoot']:.1f}, qualité {x['qual']:.3f}, {x['pos']})")
    true_rate = {p: x["shoot"] * x["toi"] / 20 * x["qual"] * x["finish"] for p, x in players.items()}
    pred = {int(k): v["prob"] for k, v in data["players"].items()}
    common = list(pred)
    corr = np.corrcoef([pred[p] for p in common], [true_rate[p] for p in common])[0, 1]
    print(f"\nCorrélation prédiction / vrai talent : {corr:.2f}")
    c0 = top[0]
    print("Facteurs du n°1 :", [(f["param"], f["mult"]) for f in c0["factors"]][:8], "…")

    print("=== RÉSULTATS ===")
    S.boxscore_goals = lambda gid: {int(k): int(rng.random() < v["prob"]) for k, v in data["players"].items()}
    results.MIN_NIGHTS_LIVE = 1
    results.grade()
    results.summarize()
    print(json.loads((C.DOCS / "precision.json").read_text())["source"])


if __name__ == "__main__":
    main()
