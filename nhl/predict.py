"""Prédictions de la soirée NHL (3 passages par jour).

    python -m nhl.predict --stage preview   # 9h Paris : première estimation
    python -m nhl.predict --stage update    # 18h : blessures, gardiens probables
    python -m nhl.predict --stage final     # minuit : version archivée et évaluée
"""
import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as C
from . import sources as S
from .features import FEATURE_LIST, build
from .model import load, predict
from .train import load_history

BOOK_WEIGHT = 0.5      # 0 = on ignore les bookmakers, 1 = on les suit totalement
TOP_GLOBAL, TOP_GAME = 25, 10


def night():
    return datetime.now(ZoneInfo("America/New_York")).date()


# ------------------------------------------------------------ alignements

def lineups(sk, games):
    """Alignement probable : joueurs du dernier match de l'équipe encore dans l'effectif."""
    allsit = sk[sk["situation"] == "all"]
    out = {}
    for gm in games:
        for team in (gm["home"], gm["away"]):
            d = allsit[allsit["team"] == team]
            if d.empty:
                out[team] = pd.DataFrame()
                continue
            last = d[d["date"] == d["date"].max()]
            ros = S.roster(team)
            if ros:
                last = last[last["player_id"].isin(ros)]
            out[team] = last[["player_id", "name", "pos"]].drop_duplicates("player_id")
    return out


def projected_starter(team, day):
    """Gardien probable : celui qui joue le plus, sauf s'il a joué la veille."""
    starts = S.goalie_starts(team, n=10)
    if not starts:
        return None
    counts = {}
    for s in starts:
        counts.setdefault(s["id"], {"n": 0, "name": s["name"]})["n"] += 1
    order = sorted(counts.items(), key=lambda kv: -kv[1]["n"])
    yesterday = (day - timedelta(days=1)).isoformat()
    pick = order[0]
    if starts[-1]["date"] == yesterday and starts[-1]["id"] == pick[0] and len(order) > 1:
        pick = order[1]                     # le n°1 a joué hier → back-to-back, on prévoit le remplaçant
    return {"id": pick[0], "name": pick[1]["name"], "status": "probable",
            "share": round(pick[1]["n"] / len(starts), 2)}


# ------------------------------------------------------------ helpers

def lam(p):
    return -math.log(max(1e-6, 1 - p))


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def mood(row):
    """Badge 🔥 / ❄️ : ce que montrent ses tirs, sa glace et son rôle en PP."""
    def v(c):
        x = row.get(c)
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    score, why = 0, []
    fen, toi, pp = v("f_fen_trend"), v("f_toi_trend"), v("f_pp_trend")
    if fen is not None and fen > 0.2:
        score += 1; why.append(f"tire {round(fen * 100)} % de plus que d'habitude")
    if fen is not None and fen < -0.25:
        score -= 1; why.append(f"tire {round(-fen * 100)} % de moins que d'habitude")
    if toi is not None and toi > 1.5:
        score += 1; why.append(f"+{toi:.1f} min de glace sur ses 3 derniers matchs".replace(".", ","))
    if toi is not None and toi < -1.5:
        score -= 1; why.append(f"{toi:.1f} min de glace sur ses 3 derniers matchs".replace(".", ","))
    if pp is not None and pp > 0.7:
        score += 1; why.append("plus utilisé en supériorité numérique")
    if pp is not None and pp < -0.7:
        score -= 1; why.append("moins utilisé en supériorité numérique")
    st, dr = v("f_goal_streak") or 0, v("f_drought") or 0
    if st >= 2:
        why.append(f"a marqué lors de ses {int(st)} derniers matchs")
    if dr >= 8:
        why.append(f"{int(dr)} matchs sans but")
    if (v("f_since_return") or 10) < 3:
        why.append("revient de blessure")
    if v("f_vs_former"):
        why.append("affronte son ancien club")
    m = "hot" if score >= 2 else "cold" if score <= -2 else None
    return {"mood": m, "why": why}


def player_card(row, pred, hist, book_mult, rank):
    p0 = pred["p"]
    l = lam(p0) * book_mult
    p = 1 - math.exp(-l)
    factors = pred["factors"] + ([{"group": "book", "param": "Total de buts de l'équipe",
                                   "mult": round(book_mult, 3)}] if book_mult != 1 else [])
    h = hist.sort_values("date").tail(10)
    season = hist[hist["season"] == C.CURRENT_SEASON]
    gp = max(1, len(season))
    shots = (season["hd"].sum(), season["md"].sum(), season["ld"].sum())
    tot = sum(shots) or 1
    return {
        "id": int(row["player_id"]), "name": row["name"], "pos": row["pos"], "rank": rank,
        "team": row["team"], "opp": row["opp"], "home": bool(row["home"]),
        "prob": round(p, 4), "p2": round(1 - math.exp(-l) - l * math.exp(-l), 4), "xg": round(l, 3),
        "base": round(pred["base"], 4), "factors": factors,
        "detail": {
            "goalie": row.get("goalie_name"), "goalie_gsax": r(row.get("g_gsax60")),
            "opp_xga": r(row.get("o_xga_pg")), "opp_pen": r(row.get("o_pen_pg")),
            "opp_blocks": r(row.get("o_blocks_pg"), 1), "toi": r(row.get("f_toi_l3"), 1),
            "pp_toi": r(row.get("f_pp_toi"), 1), "onice": r(row.get("f_onice_xgf60")),
            "finish": r(row.get("f_finish"), 3), "rest": r(row.get("t_rest"), 0),
            "opp_rest": r(row.get("o_rest"), 0), "travel": r(row.get("t_travel"), 0),
            "missing": r(row.get("t_missing_toi"), 1), "h2h": r(row.get("f_h2h"), 3),
        },
        "last10": [{"xg": r(x), "g": int(g)} for x, g in zip(h["ixg"], h["goals"])],
        "stats": {
            "gp": int(len(season)), "goals": int(season["goals"].sum()),
            "sog_pg": r(season["sog"].sum() / gp, 1), "fen_pg": r(season["fenwick"].sum() / gp, 1),
            "toi": r(season["toi"].mean() / 60 if len(season) else None, 1),
            "pp_toi": r(row.get("f_pp_toi"), 1),
            "sh_pct": r(season["goals"].sum() / max(1, season["sog"].sum()) * 100, 1),
            "hd_pg": r(season["hd"].sum() / gp, 1), "xg_pg": r(season["ixg"].sum() / gp),
        },
        "zones": [round(s / tot * 100) for s in shots],
        **mood(row),
    }


# ------------------------------------------------------------ main

def run(day, stage):
    games = S.schedule(day)
    print(f"Soirée du {day} · {len(games)} matchs · passage « {stage} »")
    payload = {"date": day.isoformat(), "stage": stage,
               "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "games": [], "top": []}
    if not games:
        write(payload, stage)
        return
    seasons = [C.CURRENT_SEASON - 2, C.CURRENT_SEASON - 1, C.CURRENT_SEASON]
    sk, gl, tm = load_history(seasons)
    lu = lineups(sk, games)
    books = S.team_totals()

    up, starters, goalies = [], [], {}
    for gm in games:
        gm["date"] = day.isoformat()
        for team, opp, home in ((gm["home"], gm["away"], 1), (gm["away"], gm["home"], 0)):
            for _, p in lu.get(team, pd.DataFrame()).iterrows():
                up.append({"player_id": p["player_id"], "name": p["name"], "pos": p["pos"], "team": team,
                           "opp": opp, "home": home, "date": pd.Timestamp(day), "game_id": gm["game_id"],
                           "season": C.CURRENT_SEASON})
            st = projected_starter(team, day)
            goalies[(gm["game_id"], team)] = st
            if st:
                starters.append({"game_id": gm["game_id"], "team": team, "player_id": st["id"],
                                 "name": st["name"], "date": day.isoformat(), "situation": "all"})

    df = build(sk, gl, tm, pd.DataFrame(up), games, starters)
    fut = df[df["future"]].reset_index(drop=True)
    booster, iso = load()
    if booster.feature_name() != FEATURE_LIST:
        raise SystemExit("Le modèle ne correspond plus aux paramètres du code : lance « train ».")
    preds = predict(booster, iso, fut)

    # Ajustement bookmakers au niveau de l'équipe.
    fut["lam"] = [lam(x["p"]) for x in preds]
    team_lam = fut.groupby("team")["lam"].sum().to_dict()
    mult = {t: (max(0.8, min(1.25, books[t] / v)) ** BOOK_WEIGHT if t in books and v > 0 else 1.0)
            for t, v in team_lam.items()}

    hist = df[~df["future"]]
    by_pid = dict(tuple(hist.groupby("player_id")))
    cards = [player_card(fut.iloc[i], preds[i], by_pid.get(fut.iloc[i]["player_id"], hist.iloc[:0]),
                         mult.get(fut.iloc[i]["team"], 1.0), 0) for i in range(len(fut))]
    cards.sort(key=lambda c: -c["prob"])
    for i, c in enumerate(cards):
        c["rank"] = i + 1

    tctx = df[df["future"]].drop_duplicates("team").set_index("team")
    for gm in games:
        side = {}
        for t in (gm["away"], gm["home"]):
            xg_team = sum(c["xg"] for c in cards if c["team"] == t)
            row = tctx.loc[t] if t in tctx.index else {}
            side[t] = {"abbrev": t, "name": gm["home_name"] if t == gm["home"] else gm["away_name"],
                       "xg": round(xg_team, 2), "book": books.get(t),
                       "rest": r(row.get("t_rest"), 0),
                       "goalie": goalies.get((gm["game_id"], t))}
        # Victoire : Poisson sur les buts attendus des deux équipes (approximation).
        a, h = side[gm["away"]]["xg"], side[gm["home"]]["xg"]
        pw = win_prob(h, a)
        payload["games"].append({
            "id": gm["game_id"], "start": gm["start"], "state": gm["state"], "venue": gm["venue"],
            "away": side[gm["away"]], "home": side[gm["home"]], "home_win": round(pw, 3),
            "ctx": team_compare(df, gm),
            "players": [c["id"] for c in cards if c["team"] in (gm["home"], gm["away"])][:TOP_GAME],
        })
    payload["players"] = {c["id"]: c for c in cards}
    payload["top"] = [c["id"] for c in cards[:TOP_GLOBAL]]
    payload["books"] = bool(books)
    write(payload, stage)


def win_prob(h, a):
    """P(victoire domicile) avec Poisson + prolongation partagée."""
    ph = [math.exp(-h) * h ** k / math.factorial(k) for k in range(12)]
    pa = [math.exp(-a) * a ** k / math.factorial(k) for k in range(12)]
    win = sum(ph[i] * pa[j] for i in range(12) for j in range(12) if i > j)
    tie = sum(ph[i] * pa[i] for i in range(12))
    return win + tie * 0.52


def team_compare(df, gm):
    """Tableau comparatif équipe contre équipe (20 derniers matchs)."""
    fut = df[df["future"] & df["game_id"].eq(gm["game_id"])].drop_duplicates("team").set_index("team")
    def v(t, c, n=2):
        return r(fut.loc[t, c], n) if t in fut.index and c in fut.columns else None
    a, h = gm["away"], gm["home"]
    return [
        {"label": "xG pour / match", "away": v(a, "t_xgf_pg"), "home": v(h, "t_xgf_pg"), "better": "high"},
        {"label": "xG contre / match", "away": v(h, "o_xga_pg"), "home": v(a, "o_xga_pg"), "better": "low"},
        {"label": "Pénalités / match", "away": v(h, "o_pen_pg"), "home": v(a, "o_pen_pg"), "better": "low"},
        {"label": "Jours de repos", "away": v(a, "t_rest", 0), "home": v(h, "t_rest", 0), "better": "high"},
        {"label": "Déplacement (km)", "away": v(a, "t_travel", 0), "home": v(h, "t_travel", 0), "better": "low"},
    ]


def write(payload, stage):
    C.DOCS.mkdir(exist_ok=True)
    txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    (C.DOCS / "data.json").write_text(txt)
    if stage == "final":
        C.HISTORY.mkdir(exist_ok=True)
        (C.HISTORY / f"{payload['date']}.json").write_text(txt)
    print(f"→ {len(payload.get('top', []))} joueurs au top, écrit dans docs/data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preview", "update", "final"], default="update")
    ap.add_argument("--date", help="AAAA-MM-JJ (par défaut : la soirée NHL en cours)")
    a = ap.parse_args()
    run(date.fromisoformat(a.date) if a.date else night(), a.stage)


if __name__ == "__main__":
    main()
