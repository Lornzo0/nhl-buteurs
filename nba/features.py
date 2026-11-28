"""Paramètres du modèle NBA (probabilité de marquer 20 points ou plus).

Même principe que pour la NHL : le même code calcule les paramètres pour
l'historique et pour les matchs à venir, uniquement à partir des matchs
PRÉCÉDENTS (shift(1)).
"""
import numpy as np
import pandas as pd

from nhl.features import _prev, _run
from . import config as C
from .sources import team_logs

# feature -> (famille, paramètre affiché dans l'app)
FEATURES = {
    "f_pts_ewm":      ("player", "Niveau habituel"),
    "f_pts_season":   ("player", "Niveau habituel"),
    "f_fga_ewm":      ("player", "Volume de tirs"),
    "f_fg3a_ewm":     ("player", "Volume de tirs"),
    "f_fta_ewm":      ("player", "Lancers francs provoqués"),
    "f_usage":        ("player", "Rôle offensif"),
    "f_share":        ("player", "Rôle offensif"),
    "f_pts_l5":       ("player", "Forme récente"),
    "f_ts":           ("player", "Adresse"),
    "f_min_ewm":      ("player", "Minutes projetées"),
    "f_min_l3":       ("player", "Minutes projetées"),
    "f_p20_rate":     ("player", "Régularité à 20+"),
    "f_pts_sd":       ("player", "Régularité à 20+"),
    "f_starter":      ("player", "Titulaire"),
    "f_games":        ("player", "Expérience"),
    "f_streak20":     ("player", "Confiance & séries"),
    "f_min_trend":    ("player", "Confiance & séries"),
    "f_fga_trend":    ("player", "Confiance & séries"),
    "o_drtg":         ("opp", "Défense adverse"),
    "o_pace":         ("opp", "Rythme de l'adversaire"),
    "t_pace":         ("ctx", "Rythme de l'équipe"),
    "home":           ("ctx", "Domicile / extérieur"),
    "t_rest":         ("ctx", "Repos / back-to-back"),
    "o_rest":         ("ctx", "Repos / back-to-back"),
    "t_travel":       ("ctx", "Déplacement"),
    "t_missing_fga":  ("ctx", "Absences dans l'équipe"),
    "t_strength":     ("ctx", "Écart de niveau (risque de blowout)"),
    "f_h2h":          ("ctx", "Historique face-à-face"),
    "f_since_return": ("ctx", "Retour de blessure"),
    "f_vs_former":    ("ctx", "Match contre son ancien club"),
}
FEATURE_LIST = list(FEATURES)
# Bon sens : plus de points habituels / de minutes / une défense adverse plus
# perméable ne peuvent pas faire BAISSER la probabilité.
MONOTONE = {"f_pts_ewm": 1, "f_min_l3": 1, "o_drtg": 1, "f_usage": 1}


def _km(a, b):
    import math
    if a not in C.ARENAS or b not in C.ARENAS:
        return 0.0
    (la1, lo1), (la2, lo2) = C.ARENAS[a], C.ARENAS[b]
    p = math.pi / 180
    h = (math.sin((la2 - la1) * p / 2) ** 2
         + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(h))


# ------------------------------------------------------------------ joueurs

def player_table(pl, upcoming=None):
    df = pl.copy()
    df["future"] = False
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["future"] = True
        df = pd.concat([df, up], ignore_index=True)
    return df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)


def add_player_features(df):
    g = df.groupby("player_id", sort=False)
    ewm = lambda h: (lambda s: s.ewm(halflife=h, min_periods=1).mean())
    roll = lambda n: (lambda s: s.rolling(n, min_periods=1).mean())

    df["f_pts_ewm"] = _prev(g, "pts", ewm(10))
    df["f_pts_l5"] = _prev(g, "pts", roll(5))
    df["f_fga_ewm"] = _prev(g, "fga", ewm(10))
    df["f_fg3a_ewm"] = _prev(g, "fg3a", ewm(10))
    df["f_fta_ewm"] = _prev(g, "fta", ewm(10))
    df["f_min_ewm"] = _prev(g, "min", ewm(8))
    df["f_min_l3"] = _prev(g, "min", roll(3))
    df["f_games"] = g.cumcount()
    df["f_starter"] = _prev(g, "starter", lambda s: s.rolling(5, min_periods=1).mean())

    # Usage : tirs + lancers + pertes de balle par minute jouée.
    df["_use"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    su = _prev(g, "_use", lambda s: s.rolling(15, min_periods=1).sum())
    sm = _prev(g, "min", lambda s: s.rolling(15, min_periods=1).sum())
    df["f_usage"] = su / sm.replace(0, np.nan)

    # Adresse : « true shooting » sur 40 matchs, régressé vers 57 % (100 tentatives fictives).
    sp = _prev(g, "pts", lambda s: s.rolling(40, min_periods=1).sum())
    sa = _prev(g, "fga", lambda s: s.rolling(40, min_periods=1).sum())
    sf = _prev(g, "fta", lambda s: s.rolling(40, min_periods=1).sum())
    att = 2 * (sa + 0.44 * sf)
    df["f_ts"] = (sp + 0.57 * 100) / (att + 100)

    # Régularité : part de ses 20 derniers matchs à 20+ (régressée) et dispersion des points.
    df["_p20"] = (df["pts"] >= C.THRESHOLD).astype(float)
    df.loc[df["future"], "_p20"] = np.nan
    n20 = _prev(g, "_p20", lambda s: s.rolling(20, min_periods=1).count())
    c20 = _prev(g, "_p20", lambda s: s.rolling(20, min_periods=1).sum())
    df["f_p20_rate"] = (c20.fillna(0) + 0.1 * 5) / (n20.fillna(0) + 5)
    df["f_pts_sd"] = _prev(g, "pts", lambda s: s.rolling(20, min_periods=5).std())

    gs = df.groupby(["player_id", "season"], sort=False)
    df["f_pts_season"] = gs["pts"].transform(lambda s: s.shift(1).expanding().mean())
    df["f_pts_season"] = df["f_pts_season"].fillna(df["f_pts_ewm"])

    # Face-à-face : points contre cet adversaire au-delà de son niveau, régressé.
    gh = df.groupby(["player_id", "opp"], sort=False)
    hp = gh["pts"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    hn = gh.cumcount()
    lvl = df["f_pts_ewm"].fillna(8)
    df["f_h2h"] = (hp + 4 * lvl) / (hn + 4) - lvl

    pid = df["player_id"]
    df["f_streak20"] = _run(g["_p20"].shift(1) >= 1, pid)
    fga_l5 = _prev(g, "fga", lambda s: s.rolling(5, min_periods=3).mean())
    df["f_fga_trend"] = fga_l5 / df["f_fga_ewm"].replace(0, np.nan) - 1
    df["f_min_trend"] = df["f_min_l3"] - df["f_min_ewm"]

    # Retour de blessure : matchs depuis une absence de 10 jours ou plus.
    gap = g["date"].diff().dt.days
    back = (gap >= 10) & g["season"].diff().eq(0)
    seg = back.astype(int).groupby(pid).cumsum()
    since = df.groupby([pid, seg]).cumcount()
    df["f_since_return"] = np.where(seg > 0, since, 10).clip(max=10)

    first = (df[~df["future"]].groupby(["player_id", "team"])["date"].min()
             .rename("first_with_opp").reset_index().rename(columns={"team": "opp"}))
    m = df[["player_id", "opp", "date"]].merge(first, on=["player_id", "opp"], how="left")
    df["f_vs_former"] = (m["first_with_opp"].notna() & (m["first_with_opp"] < m["date"])).astype(int).to_numpy()
    return df.drop(columns=["_use", "_p20"])


def team_context(df):
    """Absences : tirs habituels des joueurs réguliers absents, et part des tirs du joueur."""
    miss, share = {}, {}
    for team, d in df.groupby("team", sort=False):
        games = d.drop_duplicates("game_id").sort_values("date")["game_id"].tolist()
        present = d.groupby("game_id")["player_id"].apply(set).to_dict()
        fga = d.set_index(["game_id", "player_id"])["f_fga_ewm"].to_dict()
        for i, gid in enumerate(games):
            recent = games[max(0, i - 5):i]
            if len(recent) < 3:
                miss[(gid, team)] = 0.0
                continue
            counts = {}
            for r in recent:
                for p in present.get(r, ()):
                    counts[p] = counts.get(p, 0) + 1
            regulars = {p for p, n in counts.items() if n >= 3}
            gone = regulars - present.get(gid, set())
            miss[(gid, team)] = sum(fga.get((recent[-1], p), 0) or 0 for p in gone)
        tot = d.groupby("game_id")["f_fga_ewm"].sum().to_dict()
        for gid, p, v in zip(d["game_id"], d["player_id"], d["f_fga_ewm"]):
            share[(gid, p)] = (v or 0) / tot[gid] if tot.get(gid) else np.nan
    df["t_missing_fga"] = [miss.get((g, t), 0.0) for g, t in zip(df["game_id"], df["team"])]
    df["f_share"] = [share.get((g, p), np.nan) for g, p in zip(df["game_id"], df["player_id"])]
    return df


# ------------------------------------------------------------------ équipes

def team_table(pl, upcoming_games=None):
    t = team_logs(pl)
    t["future"] = False
    if upcoming_games:
        rows = []
        for gm in upcoming_games:
            for team, opp, home in ((gm["home"], gm["away"], 1), (gm["away"], gm["home"], 0)):
                rows.append({"game_id": gm["game_id"], "date": pd.Timestamp(gm["date"]),
                             "season": C.CURRENT_SEASON, "team": team, "opp": opp, "home": home, "future": True})
        t = pd.concat([t, pd.DataFrame(rows)], ignore_index=True)
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    g = t.groupby("team", sort=False)
    roll = lambda s: s.rolling(15, min_periods=1).sum()
    poss = _prev(g, "poss", roll)
    t["t_ortg"] = _prev(g, "pts", roll) / poss * 100
    t["t_drtg"] = _prev(g, "opp_pts", roll) / poss * 100
    t["t_pace"] = _prev(g, "poss", lambda s: s.rolling(15, min_periods=1).mean())
    t["t_ppg"] = _prev(g, "pts", lambda s: s.rolling(15, min_periods=1).mean())
    t["t_rest"] = g["date"].diff().dt.days.clip(upper=4).fillna(3)
    t["arena"] = np.where(t["home"] == 1, t["team"], t["opp"])
    t["prev_arena"] = g["arena"].shift(1)
    t["t_travel"] = [_km(a, b) if isinstance(b, str) else 0.0 for a, b in zip(t["arena"], t["prev_arena"])]
    o = t[["game_id", "team", "t_ortg", "t_drtg", "t_pace", "t_rest"]].rename(columns={
        "team": "opp", "t_ortg": "o_ortg", "t_drtg": "o_drtg", "t_pace": "o_pace", "t_rest": "o_rest"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["t_strength"] = (t["t_ortg"] - t["t_drtg"]) - (t["o_ortg"] - t["o_drtg"])
    return t[["game_id", "team", "t_ortg", "t_drtg", "t_pace", "t_ppg", "t_rest", "t_travel",
              "o_ortg", "o_drtg", "o_pace", "o_rest", "t_strength"]]


# ------------------------------------------------------------------ assemblage

def build(pl, upcoming=None, upcoming_games=None):
    df = add_player_features(player_table(pl, upcoming))
    df = team_context(df)
    df = df.merge(team_table(pl, upcoming_games), on=["game_id", "team"], how="left")
    df["y"] = (df["pts"] >= C.THRESHOLD).astype(float)
    df.loc[df["future"], "y"] = np.nan
    return df
