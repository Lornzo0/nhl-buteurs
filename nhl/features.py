"""Construction des paramètres (features) du modèle.

Principe clé : le MÊME code calcule les paramètres pour l'historique
(entraînement) et pour les matchs à venir (prédiction). Les matchs à venir
sont ajoutés comme des lignes sans résultat, puis chaque paramètre est calculé
uniquement à partir des matchs PRÉCÉDENTS (shift(1)) : aucune information du
match lui-même ne fuit dans sa prédiction.
"""
import math

import numpy as np
import pandas as pd

from . import config as C

# feature -> (famille, paramètre affiché dans l'app)
FEATURES = {
    "f_fen_pg":       ("player", "Volume de tirs"),
    "f_sog_pg":       ("player", "Volume de tirs"),
    "f_xg_per_shot":  ("player", "Qualité des tirs"),
    "f_hd_pg":        ("player", "Qualité des tirs"),
    "f_ixg_season":   ("player", "Niveau habituel"),
    "f_ixg_ewm":      ("player", "Niveau habituel"),
    "f_ixg_l5":       ("player", "Forme récente"),
    "f_finish":       ("player", "Talent de finition"),
    "f_toi_l3":       ("player", "Temps de glace projeté"),
    "f_toi_ewm":      ("player", "Temps de glace projeté"),
    "f_pp_toi":       ("player", "Supériorité numérique"),
    "f_pp_ixg":       ("player", "Supériorité numérique"),
    "f_onice_xgf60":  ("player", "Ligne & coéquipiers"),
    "f_is_d":         ("player", "Poste & expérience"),
    "f_games":        ("player", "Poste & expérience"),
    "g_gsax60":       ("opp", "Gardien probable"),
    "g_rest":         ("opp", "Fatigue du gardien"),
    "o_xga_pg":       ("opp", "Défense adverse"),
    "o_pen_pg":       ("opp", "Discipline adverse"),
    "o_blocks_pg":    ("opp", "Tirs bloqués adverses"),
    "home":           ("ctx", "Domicile / extérieur"),
    "t_rest":         ("ctx", "Repos / back-to-back"),
    "o_rest":         ("ctx", "Repos / back-to-back"),
    "t_travel":       ("ctx", "Déplacement"),
    "t_missing_toi":  ("ctx", "Absences dans l'équipe"),
    "t_strength":     ("ctx", "Filet désert & rapport de force"),
    "f_h2h":          ("ctx", "Historique face-à-face"),
    # Confiance : séries, disettes, et surtout ce que fait l'entraîneur.
    "f_goal_streak":  ("player", "Confiance & séries"),
    "f_drought":      ("player", "Confiance & séries"),
    "f_pts_streak":   ("player", "Confiance & séries"),
    "f_fen_trend":    ("player", "Confiance & séries"),
    "f_toi_trend":    ("player", "Confiance & séries"),
    "f_pp_trend":     ("player", "Confiance & séries"),
    "f_since_return": ("ctx", "Retour de blessure"),
    "f_vs_former":    ("ctx", "Match contre son ancien club"),
}
FEATURE_LIST = list(FEATURES)
# Contraintes de bon sens : plus de tirs / de glace / une défense adverse plus
# perméable ne peuvent pas faire BAISSER la probabilité, un meilleur gardien
# adverse ne peut pas la faire MONTER.
MONOTONE = {"f_fen_pg": 1, "f_ixg_ewm": 1, "f_toi_l3": 1, "o_xga_pg": 1, "g_gsax60": -1}


def _km(a, b):
    if a not in C.ARENAS or b not in C.ARENAS:
        return 0.0
    (la1, lo1), (la2, lo2) = C.ARENAS[a], C.ARENAS[b]
    p = math.pi / 180
    h = (math.sin((la2 - la1) * p / 2) ** 2
         + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(h))


def _prev(g, col, fn):
    """Applique fn sur la série décalée d'un match, groupe par groupe."""
    return g[col].transform(lambda s: fn(s.shift(1)))


# ------------------------------------------------------------------ joueurs

def player_table(sk, upcoming=None):
    """Une ligne par joueur et par match, avec les stats du match (cible)."""
    base = sk[sk["situation"] == "all"].copy()
    pp = (sk[sk["situation"] == "5on4"][["player_id", "game_id", "toi", "ixg"]]
          .rename(columns={"toi": "pp_toi", "ixg": "pp_ixg"}))
    df = base.merge(pp, on=["player_id", "game_id"], how="left")
    df[["pp_toi", "pp_ixg"]] = df[["pp_toi", "pp_ixg"]].fillna(0)
    df["toi"] = df["toi"] / 60          # secondes → minutes
    df["pp_toi"] = df["pp_toi"] / 60
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

    df["f_fen_pg"] = _prev(g, "fenwick", ewm(15))
    df["f_sog_pg"] = _prev(g, "sog", ewm(15))
    df["f_hd_pg"] = _prev(g, "hd", ewm(15))
    df["f_ixg_ewm"] = _prev(g, "ixg", ewm(10))
    df["f_ixg_l5"] = _prev(g, "ixg", roll(5))
    df["f_toi_ewm"] = _prev(g, "toi", ewm(10))
    df["f_toi_l3"] = _prev(g, "toi", roll(3))
    df["f_pp_toi"] = _prev(g, "pp_toi", ewm(10))
    df["f_pp_ixg"] = _prev(g, "pp_ixg", ewm(10))
    df["f_games"] = g.cumcount()

    # Qualité de tir : xG par tentative non bloquée, lissé.
    sx = _prev(g, "ixg", lambda s: s.rolling(60, min_periods=1).sum())
    sf = _prev(g, "fenwick", lambda s: s.rolling(60, min_periods=1).sum())
    df["f_xg_per_shot"] = (sx + 0.07 * 20) / (sf + 20)      # régressé vers 0,07

    # Moyenne de la saison en cours.
    gs = df.groupby(["player_id", "season"], sort=False)
    df["f_ixg_season"] = gs["ixg"].transform(lambda s: s.shift(1).expanding().mean())
    df["f_ixg_season"] = df["f_ixg_season"].fillna(df["f_ixg_ewm"])

    # Talent de finition : buts − xG sur ~3 saisons, régressé (le +15 xG
    # ramène vers 0 les joueurs sur un petit échantillon).
    rg = _prev(g, "goals", lambda s: s.rolling(246, min_periods=1).sum())
    rx = _prev(g, "ixg", lambda s: s.rolling(246, min_periods=1).sum())
    df["f_finish"] = (rg - rx) / (rx + 15)

    # Coéquipiers : xG de l'équipe quand il est sur la glace, par 60 min.
    ox = _prev(g, "onice_xgf", lambda s: s.rolling(20, min_periods=1).sum())
    ot = _prev(g, "toi", lambda s: s.rolling(20, min_periods=1).sum())
    df["f_onice_xgf60"] = ox / ot.replace(0, np.nan) * 60

    df["f_is_d"] = (df["pos"].astype(str).str.upper() == "D").astype(int)

    # Face-à-face : buts contre cet adversaire au-delà de son niveau attendu.
    gh = df.groupby(["player_id", "opp"], sort=False)
    hg = gh["goals"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    hn = gh.cumcount()
    rate = df["f_ixg_ewm"].fillna(0.1)
    df["f_h2h"] = (hg + 5 * rate) / (hn + 5) - rate
    add_confidence(df, g)
    return df


def _run(prev_flag, keys):
    """Longueur de la série en cours (matchs consécutifs où prev_flag est vrai)."""
    f = prev_flag.fillna(False).astype(bool)
    block = (~f).groupby(keys).cumsum()
    return f.groupby([keys, block]).cumsum()


def add_confidence(df, g):
    pid = df["player_id"]
    prev_goals = g["goals"].shift(1)
    df["f_goal_streak"] = _run(prev_goals >= 1, pid)
    df["f_drought"] = _run(prev_goals == 0, pid).clip(upper=30)
    if "points" in df and df["points"].notna().any():
        df["f_pts_streak"] = _run(g["points"].shift(1) >= 1, pid)
    else:
        df["f_pts_streak"] = np.nan

    # Tendances : il tire plus ? l'entraîneur lui donne plus de glace / de PP ?
    fen_l5 = _prev(g, "fenwick", lambda s: s.rolling(5, min_periods=3).mean())
    df["f_fen_trend"] = fen_l5 / df["f_fen_pg"].replace(0, np.nan) - 1
    df["f_toi_trend"] = df["f_toi_l3"] - df["f_toi_ewm"]
    pp_l3 = _prev(g, "pp_toi", lambda s: s.rolling(3, min_periods=1).mean())
    df["f_pp_trend"] = pp_l3 - df["f_pp_toi"]

    # Retour de blessure : nombre de matchs depuis une absence de 12 jours ou plus.
    gap = g["date"].diff().dt.days
    same_season = g["season"].diff().eq(0)
    back = (gap >= 12) & same_season
    seg = back.astype(int).groupby(pid).cumsum()
    since = df.groupby([pid, seg]).cumcount()
    df["f_since_return"] = np.where(seg > 0, since, 10).clip(max=10)

    # Ancien club : a-t-il déjà joué POUR l'équipe qu'il affronte ?
    first = (df[~df["future"]].groupby(["player_id", "team"])["date"].min()
             .rename("first_with_opp").reset_index().rename(columns={"team": "opp"}))
    m = df[["player_id", "opp", "date"]].merge(first, on=["player_id", "opp"], how="left")
    df["f_vs_former"] = (m["first_with_opp"].notna() & (m["first_with_opp"] < m["date"])).astype(int).to_numpy()


# ------------------------------------------------------------------ équipes

def team_table(tm, upcoming_games=None):
    t = tm[["game_id", "date", "season", "team", "opp", "home", "xgf", "xga", "pen", "blocked_against"]].copy()
    t["future"] = False
    if upcoming_games:
        rows = []
        for gm in upcoming_games:
            for team, opp, home in ((gm["home"], gm["away"], 1), (gm["away"], gm["home"], 0)):
                rows.append({"game_id": gm["game_id"], "date": pd.Timestamp(gm["date"]),
                             "season": C.CURRENT_SEASON, "team": team, "opp": opp,
                             "home": home, "future": True})
        t = pd.concat([t, pd.DataFrame(rows)], ignore_index=True)
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    g = t.groupby("team", sort=False)
    roll = lambda s: s.rolling(20, min_periods=1).mean()
    for src, dst in (("xgf", "t_xgf_pg"), ("xga", "t_xga_pg"), ("pen", "t_pen_pg"),
                     ("blocked_against", "t_blocks_pg")):
        t[dst] = _prev(g, src, roll)
    t["t_rest"] = g["date"].diff().dt.days.clip(upper=5).fillna(3)
    t["arena"] = np.where(t["home"] == 1, t["team"], t["opp"])
    t["prev_arena"] = g["arena"].shift(1)
    t["t_travel"] = [_km(a, b) if isinstance(b, str) else 0.0 for a, b in zip(t["arena"], t["prev_arena"])]

    o = t[["game_id", "team", "t_xgf_pg", "t_xga_pg", "t_pen_pg", "t_blocks_pg", "t_rest"]].rename(columns={
        "team": "opp", "t_xgf_pg": "o_xgf_pg", "t_xga_pg": "o_xga_pg", "t_pen_pg": "o_pen_pg",
        "t_blocks_pg": "o_blocks_pg", "t_rest": "o_rest"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["t_strength"] = (t["t_xgf_pg"] - t["t_xga_pg"]) - (t["o_xgf_pg"] - t["o_xga_pg"])
    return t[["game_id", "team", "t_rest", "t_travel", "t_strength", "o_xga_pg",
              "o_pen_pg", "o_blocks_pg", "o_rest", "t_xgf_pg"]]


def missing_toi(df):
    """Temps de glace habituel des titulaires absents de l'alignement.

    Un « titulaire » a joué au moins 3 des 5 derniers matchs de l'équipe.
    Quand il manque, son temps de glace se redistribue aux autres : c'est
    une hausse d'opportunités pour ceux qui jouent.
    """
    out = {}
    for team, d in df.groupby("team", sort=False):
        games = d.drop_duplicates("game_id").sort_values("date")["game_id"].tolist()
        present = d.groupby("game_id")["player_id"].apply(set).to_dict()
        toi = d.set_index(["game_id", "player_id"])["f_toi_ewm"].to_dict()
        for i, gid in enumerate(games):
            recent = games[max(0, i - 5):i]
            if len(recent) < 3:
                out[(gid, team)] = 0.0
                continue
            counts = {}
            for r in recent:
                for p in present.get(r, ()):
                    counts[p] = counts.get(p, 0) + 1
            regulars = {p for p, n in counts.items() if n >= 3}
            miss = regulars - present.get(gid, set())
            out[(gid, team)] = sum(toi.get((recent[-1], p), 0) or 0 for p in miss)
    return out


# ------------------------------------------------------------------ gardiens

def goalie_features(gl, upcoming_starters=None):
    """(game_id, équipe du gardien) → qualité et repos du gardien titulaire."""
    g = gl[gl["situation"] == "all"].copy()
    g = g.sort_values("toi").drop_duplicates(["game_id", "team"], keep="last")   # titulaire
    g["future"] = False
    if upcoming_starters:
        up = pd.DataFrame(upcoming_starters)
        up["date"] = pd.to_datetime(up["date"])
        up["future"] = True
        g = pd.concat([g, up], ignore_index=True)
    g = g.sort_values(["player_id", "date"]).reset_index(drop=True)
    gg = g.groupby("player_id", sort=False)
    g["gsax"] = g["xga"] - g["ga"]
    s_gsax = _prev(gg, "gsax", lambda s: s.rolling(40, min_periods=1).sum()).fillna(0)
    s_toi = _prev(gg, "toi", lambda s: s.rolling(40, min_periods=1).sum()).fillna(0)
    g["g_gsax60"] = s_gsax / (s_toi / 3600 + 8)       # buts évités / match, régressé
    g["g_rest"] = gg["date"].diff().dt.days.clip(upper=5).fillna(3)
    return g[["game_id", "team", "player_id", "name", "g_gsax60", "g_rest"]].rename(
        columns={"team": "opp", "player_id": "goalie_id", "name": "goalie_name"})


# ------------------------------------------------------------------ assemblage

def build(sk, gl, tm, upcoming=None, upcoming_games=None, upcoming_starters=None):
    df = add_player_features(player_table(sk, upcoming))
    miss = missing_toi(df)
    df["t_missing_toi"] = [miss.get((g, t), 0.0) for g, t in zip(df["game_id"], df["team"])]
    df = df.merge(team_table(tm, upcoming_games), on=["game_id", "team"], how="left")
    df = df.merge(goalie_features(gl, upcoming_starters), on=["game_id", "opp"], how="left")
    df["y"] = (df["goals"] >= 1).astype(float)
    df.loc[df["future"], "y"] = np.nan
    return df
