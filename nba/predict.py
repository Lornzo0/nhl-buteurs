"""Prédictions de la soirée NBA : probabilité que chaque joueur marque 20 points ou plus.

    python -m nba.predict --stage auto      # lancé toutes les heures, décide seul
    python -m nba.predict --stage final --date 2026-11-28
"""
import argparse
import json
import math
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from scipy.stats import norm

from . import config as C
from . import sources as S
from .features import FEATURE_LIST, build
from .model import CUT, load, predict
from .train import load_history

BOOK_WEIGHT = 0.5
TOP_GLOBAL, TOP_GAME = 25, 10
HOME_EDGE = 0.012          # ~1,4 point d'avantage à domicile sur la projection d'équipe


def night():
    return datetime.now(ZoneInfo("America/New_York")).date()


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def v(row, c):
    x = row.get(c)
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ------------------------------------------------------------ alignements

def lineups(pl, games, inj):
    """Alignement probable : joueurs vus lors des 3 derniers matchs de l'équipe,
    toujours dans cette équipe, et pas déclarés « Out » par le rapport de blessures."""
    last_team = pl.sort_values("date").groupby("player_id").tail(1).set_index("player_id")["team"]
    out = {}
    for gm in games:
        for team in (gm["home"], gm["away"]):
            d = pl[pl["team"] == team]
            if d.empty:
                out[team] = pd.DataFrame()
                continue
            recent = d.drop_duplicates("game_id").sort_values("date")["game_id"].tail(3)
            cand = d[d["game_id"].isin(recent)].drop_duplicates("player_id")
            cand = cand[cand["player_id"].map(last_team) == team]
            rows = []
            for _, p in cand.iterrows():
                st = inj.get(S.norm_name(p["name"]), "")
                if st.lower() == "out":
                    continue
                rows.append({"player_id": p["player_id"], "name": p["name"], "status": st})
            out[team] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------ joueurs

def mood(row):
    score, why = 0, []
    mt, ft = v(row, "f_min_trend"), v(row, "f_fga_trend")
    l5, ew = v(row, "f_pts_l5"), v(row, "f_pts_ewm")
    if mt is not None and mt > 3:
        score += 1; why.append(f"+{mt:.1f} min de jeu sur ses 3 derniers matchs".replace(".", ","))
    if mt is not None and mt < -3:
        score -= 1; why.append(f"{mt:.1f} min de jeu sur ses 3 derniers matchs".replace(".", ","))
    if ft is not None and ft > 0.2:
        score += 1; why.append(f"tente {round(ft * 100)} % de tirs de plus que d'habitude")
    if ft is not None and ft < -0.2:
        score -= 1; why.append(f"tente {round(-ft * 100)} % de tirs de moins que d'habitude")
    if l5 is not None and ew and l5 > ew * 1.2 and l5 - ew > 3:
        score += 1; why.append(f"{l5:.1f} pts de moyenne sur ses 5 derniers matchs".replace(".", ","))
    if l5 is not None and ew and l5 < ew * 0.8 and ew - l5 > 3:
        score -= 1; why.append(f"seulement {l5:.1f} pts de moyenne sur 5 matchs".replace(".", ","))
    st = v(row, "f_streak20") or 0
    if st >= 3:
        why.append(f"{int(st)} matchs de suite à 20+")
    if (v(row, "f_since_return") or 10) < 3:
        why.append("revient de blessure")
    if v(row, "t_missing_fga") and v(row, "t_missing_fga") >= 12:
        why.append("des coéquipiers importants sont absents")
    if v(row, "f_vs_former"):
        why.append("affronte son ancien club")
    m = "hot" if score >= 2 else "cold" if score <= -2 else None
    return {"mood": m, "why": why}


def player_card(row, pred, hist, ratio, status):
    sd = max(4.0, v(row, "f_pts_sd") or 6.0)
    p0 = min(max(pred["p"], 1e-4), 1 - 1e-4)
    mu = CUT + sd * norm.ppf(p0)                  # moyenne implicite de la loi normale
    mu2 = mu * ratio                              # ajustement bookmakers
    p = float(1 - norm.cdf((CUT - mu2) / sd))
    factors = pred["factors"] + ([{"group": "book", "param": "Points attendus de l'équipe",
                                   "mult": round((p / (1 - p)) / (p0 / (1 - p0)), 3)}] if ratio != 1 else [])
    h = hist.sort_values("date").tail(10)
    season = hist[hist["season"] == C.CURRENT_SEASON]
    gp = len(season)
    att = 2 * (season["fga"].sum() + 0.44 * season["fta"].sum())
    starter = v(row, "f_starter")
    return {
        "id": int(row["player_id"]), "name": row["name"], "rank": 0,
        "pos": "" if starter is None else "Titulaire" if starter >= 0.5 else "Remplaçant",
        "team": row["team"], "opp": row["opp"], "home": bool(row["home"]),
        "prob": round(p, 4), "p30": round(float(1 - norm.cdf((29.5 - mu2) / sd)), 4),
        "pts": round(mu2, 1), "sd": round(sd, 1), "base": round(pred["base"], 4), "factors": factors,
        "status": status or None,
        "detail": {
            "opp_drtg": r(row.get("o_drtg"), 1), "opp_pace": r(row.get("o_pace"), 1),
            "pace": r(row.get("t_pace"), 1), "min": r(row.get("f_min_l3"), 1),
            "usage": r(row.get("f_usage"), 3), "share": r(row.get("f_share"), 3),
            "ts": r(row.get("f_ts"), 3), "fta": r(row.get("f_fta_ewm"), 1),
            "p20": r(row.get("f_p20_rate"), 3), "rest": r(row.get("t_rest"), 0),
            "opp_rest": r(row.get("o_rest"), 0), "travel": r(row.get("t_travel"), 0),
            "missing": r(row.get("t_missing_fga"), 1), "h2h": r(row.get("f_h2h"), 1),
            "strength": r(row.get("t_strength"), 1), "l5": r(row.get("f_pts_l5"), 1),
        },
        "last10": [{"p": int(a), "m": round(float(b))} for a, b in zip(h["pts"], h["min"])],
        "stats": {
            "gp": gp, "pts": r(season["pts"].mean() if gp else None, 1),
            "min": r(season["min"].mean() if gp else None, 1),
            "fga": r(season["fga"].mean() if gp else None, 1),
            "fg3a": r(season["fg3a"].mean() if gp else None, 1),
            "fta": r(season["fta"].mean() if gp else None, 1),
            "ts": r(season["pts"].sum() / att * 100 if att else None, 1),
            "n20": int((season["pts"] >= C.THRESHOLD).sum()), "best": int(season["pts"].max()) if gp else None,
        },
        **mood(row),
    }


# ------------------------------------------------------------ main

def team_proj(row):
    """Points attendus d'une équipe : rythme du match × (attaque + défense adverse) / 2."""
    vals = [v(row, c) for c in ("t_pace", "o_pace", "t_ortg", "o_drtg")]
    if any(x is None for x in vals):
        return v(row, "t_ppg")
    tp, op, ortg, odrtg = vals
    return (tp + op) / 2 * (ortg + odrtg) / 2 / 100 * (1 + HOME_EDGE * (1 if row.get("home") == 1 else -1))


def run(day, stage):
    games = S.schedule(day)
    print(f"Soirée NBA du {day} · {len(games)} matchs · passage « {stage} »")
    payload = {"date": day.isoformat(), "stage": stage, "sport": "nba", "threshold": C.THRESHOLD,
               "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "games": [], "top": [], "players": {}}
    if not games:
        write(payload)
        return
    pl = load_history([C.CURRENT_SEASON - 2, C.CURRENT_SEASON - 1, C.CURRENT_SEASON])
    inj = S.injuries()
    lu = lineups(pl, games, inj)
    books = S.team_points()

    up, status = [], {}
    for gm in games:
        gm["date"] = day.isoformat()
        for team, opp, home in ((gm["home"], gm["away"], 1), (gm["away"], gm["home"], 0)):
            for _, p in lu.get(team, pd.DataFrame()).iterrows():
                status[int(p["player_id"])] = p["status"]
                up.append({"player_id": int(p["player_id"]), "name": p["name"], "team": team, "opp": opp,
                           "home": home, "date": pd.Timestamp(day), "game_id": gm["game_id"],
                           "season": C.CURRENT_SEASON})
    if not up:
        print("Aucun alignement disponible : rien à prédire.")
        write(payload)
        return

    df = build(pl, pd.DataFrame(up), games)
    fut = df[df["future"]].reset_index(drop=True)
    booster, cal = load()
    if booster.feature_name() != FEATURE_LIST:
        raise SystemExit("Le modèle NBA ne correspond plus aux paramètres du code : lance « train ».")
    preds = predict(booster, cal, fut)

    tctx = fut.drop_duplicates("team").set_index("team")
    proj = {t: team_proj(tctx.loc[t]) for t in tctx.index}
    ratio = {t: (max(0.9, min(1.1, books[t] / proj[t])) ** BOOK_WEIGHT
                 if t in books and proj.get(t) else 1.0) for t in proj}

    hist = df[~df["future"]]
    by_pid = dict(tuple(hist.groupby("player_id")))
    cards = [player_card(fut.iloc[i], preds[i], by_pid.get(fut.iloc[i]["player_id"], hist.iloc[:0]),
                         ratio.get(fut.iloc[i]["team"], 1.0), status.get(int(fut.iloc[i]["player_id"])))
             for i in range(len(fut))]
    cards.sort(key=lambda c: -c["prob"])
    for i, c in enumerate(cards):
        c["rank"] = i + 1

    for gm in games:
        side = {}
        for t in (gm["away"], gm["home"]):
            row = tctx.loc[t] if t in tctx.index else {}
            pts = proj.get(t)
            if pts and t in books:
                pts = (pts + books[t]) / 2          # moitié modèle, moitié bookmakers (BOOK_WEIGHT)
            side[t] = {"abbrev": t, "name": gm["home_name"] if t == gm["home"] else gm["away_name"],
                       "pts": r(pts, 1), "book": books.get(t), "rest": r(row.get("t_rest"), 0) if len(row) else None}
        a, h = side[gm["away"]]["pts"] or 110, side[gm["home"]]["pts"] or 110
        payload["games"].append({
            "id": gm["game_id"], "start": gm["start"], "state": gm["state"], "venue": gm["venue"],
            "away": side[gm["away"]], "home": side[gm["home"]],
            "home_win": round(float(norm.cdf((h - a) / 12.5)), 3),
            "ctx": team_compare(tctx, gm),
            "players": [c["id"] for c in cards if c["team"] in (gm["home"], gm["away"])][:TOP_GAME],
        })
    payload["players"] = {c["id"]: c for c in cards}
    payload["top"] = [c["id"] for c in cards[:TOP_GLOBAL]]
    payload["books"] = bool(books)
    payload["injuries"] = bool(inj)
    write(payload)


def team_compare(tctx, gm):
    def val(t, c, n=1):
        return r(tctx.loc[t, c], n) if t in tctx.index and c in tctx.columns else None
    a, h = gm["away"], gm["home"]
    return [
        {"label": "Points marqués / 100 poss.", "away": val(a, "t_ortg"), "home": val(h, "t_ortg"), "better": "high", "dec": 1},
        {"label": "Points encaissés / 100 poss.", "away": val(a, "t_drtg"), "home": val(h, "t_drtg"), "better": "low", "dec": 1},
        {"label": "Rythme (possessions)", "away": val(a, "t_pace"), "home": val(h, "t_pace"), "better": None, "dec": 1},
        {"label": "Jours de repos", "away": val(a, "t_rest", 0), "home": val(h, "t_rest", 0), "better": "high", "dec": 0},
        {"label": "Déplacement (km)", "away": val(a, "t_travel", 0), "home": val(h, "t_travel", 0), "better": "low", "dec": 0},
    ]


def write(payload):
    C.DOCS.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    (C.DOCS / "data.json").write_text(txt)
    if payload.get("games"):
        C.HISTORY.mkdir(parents=True, exist_ok=True)
        (C.HISTORY / f"{payload['date']}.json").write_text(txt)
    print(f"→ {len(payload.get('top', []))} joueurs au top, écrit dans docs/nba/data.json")


def _current(day):
    f = C.DOCS / "data.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("date") == day.isoformat() else None


def auto():
    """Même logique que la NHL : avant-première, mise à jour, version finale selon l'heure du premier match."""
    day = night()
    games = S.schedule(day)
    if not games:
        print(f"Pas de match NBA la nuit du {day}.")
        if not _current(day):
            write({"date": day.isoformat(), "stage": "update", "sport": "nba", "threshold": C.THRESHOLD,
                   "games": [], "top": [], "players": {},
                   "generated": datetime.now(timezone.utc).isoformat(timespec="minutes")})
        return
    now = datetime.now(timezone.utc)
    starts = [datetime.fromisoformat(g["start"].replace("Z", "+00:00")) for g in games if g.get("start")]
    if not starts:
        run(day, "update")
        return
    first = min(starts)
    hours = (first - now).total_seconds() / 3600
    stage = "preview" if hours > 10 else "update" if hours > 2.5 else "final"
    if hours <= 0:
        print(f"Premier match commencé ({first:%H:%M} UTC) : rien à faire.")
        return
    cur = _current(day)
    if cur and cur.get("generated"):
        age = (now - datetime.fromisoformat(cur["generated"])).total_seconds() / 3600
        min_age = {"preview": 99, "update": 3, "final": 1}[stage]
        if cur["stage"] == stage and age < min_age:
            print(f"Déjà à jour ({stage}, il y a {age:.1f} h).")
            return
    print(f"Premier match dans {hours:.1f} h → passage « {stage} »")
    run(day, stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["auto", "preview", "update", "final"], default="auto")
    ap.add_argument("--date", help="AAAA-MM-JJ (par défaut : la soirée NBA en cours)")
    a = ap.parse_args()
    if a.stage == "auto" and not a.date:
        auto()
    else:
        run(date.fromisoformat(a.date) if a.date else night(), "update" if a.stage == "auto" else a.stage)


if __name__ == "__main__":
    main()
