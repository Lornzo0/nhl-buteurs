"""Sources de données NBA.

  - stats.nba.com (leaguegamelog) : feuilles de match joueur par joueur, une
    requête par saison. Le site bloque parfois les serveurs cloud : dans ce cas
    on reconstruit les mêmes données à partir des feuilles de match du CDN NBA.
  - cdn.nba.com : calendrier de la saison et feuilles de match (résultats).
  - ESPN : rapport de blessures (joueurs « Out » retirés de l'alignement).
  - The Odds API (optionnel, ODDS_API_KEY) : total de points et écart par équipe.
"""
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import config as C

S = requests.Session()
S.headers["User-Agent"] = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
STATS_HEADERS = {"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
                 "Accept": "application/json, text/plain, */*", "x-nba-stats-origin": "stats",
                 "x-nba-stats-token": "true"}
NY = ZoneInfo("America/New_York")
LOG_COLS = ["player_id", "name", "team", "opp", "home", "game_id", "date", "season", "min", "pts",
            "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "reb", "ast", "tov", "starter"]


def http(url, params=None, headers=None, retries=3, timeout=60):
    for i in range(retries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == retries - 1:
                print(f"! échec {url}: {e}", file=sys.stderr)
                return None
            time.sleep(3 * (i + 1))


def fix(t):
    return C.TEAM_FIX.get(t, t)


def season_str(s):
    return f"{s}-{str(s + 1)[2:]}"


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


# ---------------------------------------------------------------- feuilles de match

def _cache_path(name):
    C.CACHE.mkdir(parents=True, exist_ok=True)
    return C.CACHE / name


def game_logs(season):
    """Une ligne par joueur et par match joué d'une saison (saison régulière + play-in + séries)."""
    p = _cache_path(f"logs_{season}.csv.gz")
    max_age = 5 if season == C.CURRENT_SEASON else 24 * 30
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age * 3600:
        return _typed(pd.read_csv(p, dtype={"game_id": str}))
    d = _stats_logs(season)
    if d is None or d.empty:
        print(f"  stats.nba.com indisponible pour {season_str(season)} : feuilles de match du CDN")
        d = _cdn_logs(season)
    if d is None or d.empty:
        if p.exists():
            print("  (on garde la copie en cache)")
            return _typed(pd.read_csv(p, dtype={"game_id": str}))
        return pd.DataFrame(columns=LOG_COLS)
    d.to_csv(p, index=False)
    return _typed(d)


def _typed(d):
    d = d.copy()
    d["game_id"] = d["game_id"].astype(str).str.zfill(10)
    d["date"] = pd.to_datetime(d["date"])
    for c in ("team", "opp"):
        d[c] = d[c].map(fix)
    return d[LOG_COLS]


def _stats_logs(season):
    frames = []
    for stype in ("Regular Season", "PlayIn", "Playoffs"):
        data = http(C.STATS + "/leaguegamelog", headers=STATS_HEADERS, timeout=45, params={
            "Counter": 0, "Direction": "ASC", "LeagueID": "00", "PlayerOrTeam": "P",
            "Season": season_str(season), "SeasonType": stype, "Sorter": "DATE"})
        if data is None:
            if stype == "Regular Season":
                return None
            continue
        rs = data["resultSets"][0]
        frames.append(pd.DataFrame(rs["rowSet"], columns=rs["headers"]))
        time.sleep(1)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    need = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN",
            "PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "OREB", "REB", "AST", "TOV"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        sys.exit(f"Colonnes stats.nba.com introuvables : {missing}\nDisponibles : {list(raw.columns)}")
    m = raw["MATCHUP"].astype(str)
    d = pd.DataFrame({
        "player_id": raw["PLAYER_ID"].astype(int), "name": raw["PLAYER_NAME"],
        "team": raw["TEAM_ABBREVIATION"], "opp": m.str.split().str[-1],
        "home": m.str.contains("vs").astype(int), "game_id": raw["GAME_ID"].astype(str),
        "date": raw["GAME_DATE"].astype(str).str[:10], "season": season,
        "min": pd.to_numeric(raw["MIN"], errors="coerce").fillna(0),
        "pts": raw["PTS"], "fga": raw["FGA"], "fgm": raw["FGM"], "fg3a": raw["FG3A"],
        "fg3m": raw["FG3M"], "fta": raw["FTA"], "ftm": raw["FTM"], "oreb": raw["OREB"],
        "reb": raw["REB"], "ast": raw["AST"], "tov": raw["TOV"], "starter": float("nan"),
    })
    return d[d["min"] > 0]


def _minutes(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", str(s or ""))
    return int(m.group(1)) + float(m.group(2)) / 60 if m else 0.0


def cdn_box(gid):
    """Feuille de match du CDN NBA : (terminé ?, lignes joueurs)."""
    data = http(C.CDN_BOX.format(gid=gid), retries=2, timeout=30)
    if not data or "game" not in data:
        return None, []
    g = data["game"]
    final = g.get("gameStatus") == 3
    et = g.get("gameEt") or ""
    day = et[:10] if et else datetime.fromisoformat(g["gameTimeUTC"].replace("Z", "+00:00")).astimezone(NY).date().isoformat()
    rows = []
    for side, other, home in (("homeTeam", "awayTeam", 1), ("awayTeam", "homeTeam", 0)):
        t, o = g[side], g[other]
        for p in t.get("players", []):
            s = p.get("statistics", {})
            mins = _minutes(s.get("minutes") or s.get("minutesCalculated"))
            if p.get("played") != "1" or mins <= 0:
                continue
            rows.append({
                "player_id": int(p["personId"]),
                "name": p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                "team": t["teamTricode"], "opp": o["teamTricode"], "home": home, "game_id": gid,
                "date": day, "min": round(mins, 1), "pts": s.get("points", 0),
                "fga": s.get("fieldGoalsAttempted", 0), "fgm": s.get("fieldGoalsMade", 0),
                "fg3a": s.get("threePointersAttempted", 0), "fg3m": s.get("threePointersMade", 0),
                "fta": s.get("freeThrowsAttempted", 0), "ftm": s.get("freeThrowsMade", 0),
                "oreb": s.get("reboundsOffensive", 0), "reb": s.get("reboundsTotal", 0),
                "ast": s.get("assists", 0), "tov": s.get("turnovers", 0),
                "starter": 1.0 if p.get("starter") == "1" else 0.0})
    return final, rows


def _cdn_logs(season):
    """Reconstruit une saison à partir des feuilles de match (une requête par match, mises en cache)."""
    box_dir = _cache_path(f"box_{season}")
    box_dir.mkdir(exist_ok=True)
    if season == C.CURRENT_SEASON:
        gids = [g["game_id"] for g in schedule_all() if g["state"] == 3]
    else:
        yy = str(season)[2:]
        gids = [f"002{yy}{i:05d}" for i in range(1, 1231)]
    rows, misses = [], 0
    for gid in gids:
        f = box_dir / f"{gid}.json"
        if f.exists():
            rows += json.loads(f.read_text())
            continue
        final, r = cdn_box(gid)
        time.sleep(0.15)
        if final is None:
            misses += 1
            if misses > 25 and not rows:
                print("  ! le CDN NBA ne répond pas non plus", file=sys.stderr)
                return None
            continue
        if final:
            f.write_text(json.dumps(r))
            rows += r
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    d["season"] = season
    return d


def team_logs(pl):
    """Totaux d'équipe par match, calculés à partir des lignes joueurs."""
    t = (pl.groupby(["game_id", "date", "season", "team", "opp", "home"], as_index=False)
         [["pts", "fga", "fta", "oreb", "tov"]].sum())
    t["poss"] = t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    o = t[["game_id", "team", "pts", "poss"]].rename(columns={"team": "opp", "pts": "opp_pts", "poss": "opp_poss"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["poss"] = (t["poss"] + t["opp_poss"]) / 2          # possessions du match (moyenne des deux équipes)
    return t.drop(columns="opp_poss")


# ---------------------------------------------------------------- calendrier

_SCHED = None


def schedule_all():
    global _SCHED
    if _SCHED is None:
        data = http(C.CDN_SCHEDULE) or {}
        out = []
        for d in data.get("leagueSchedule", {}).get("gameDates", []):
            for g in d.get("games", []):
                gid = str(g.get("gameId", ""))
                if gid[:3] not in ("002", "004", "005"):     # saison, séries, play-in
                    continue
                h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
                if not h.get("teamTricode") or not a.get("teamTricode"):
                    continue
                day = (g.get("gameDateEst") or "")[:10]
                if not day:
                    day = datetime.strptime(d["gameDate"][:10], "%m/%d/%Y").date().isoformat()
                out.append({"game_id": gid, "day": day, "start": g.get("gameDateTimeUTC"),
                            "state": g.get("gameStatus"), "home": fix(h["teamTricode"]),
                            "away": fix(a["teamTricode"]), "home_name": h.get("teamName") or h["teamTricode"],
                            "away_name": a.get("teamName") or a["teamTricode"], "venue": g.get("arenaName", "")})
        _SCHED = out
    return _SCHED


def schedule(day):
    """Matchs d'une soirée NBA (date à l'heure de New York)."""
    return [dict(g) for g in schedule_all() if g["day"] == day.isoformat()]


def boxscore_points(gid):
    """{personId: points} pour un match terminé, ou None s'il n'est pas fini / introuvable."""
    final, rows = cdn_box(gid)
    if not final:
        return None
    pts = {r["player_id"]: int(r["pts"] or 0) for r in rows}
    return pts


# ---------------------------------------------------------------- blessures

def injuries():
    """{nom normalisé: statut} d'après ESPN (« Out », « Day-To-Day »…). {} si indisponible."""
    data = http(C.ESPN_INJURIES, retries=2) or {}
    out = {}
    for team in data.get("injuries", []):
        for i in team.get("injuries", []):
            name = (i.get("athlete") or {}).get("displayName")
            if name:
                out[norm_name(name)] = i.get("status") or ""
    return out


# ---------------------------------------------------------------- cotes

def team_points():
    """Points attendus par équipe selon les bookmakers : {abbrev: points}. {} sans clé."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    data = http(C.ODDS_URL, params={"apiKey": key, "regions": "us,eu", "markets": "spreads,totals",
                                    "oddsFormat": "decimal"})
    if not data:
        return {}
    by_name = {v.lower(): k for k, v in C.TEAM_NAMES.items()}
    by_name["los angeles clippers"] = "LAC"
    out = {}
    for ev in data:
        spread, tot = {}, []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        if o.get("point") is not None:
                            spread.setdefault(o["name"], []).append(o["point"])
                elif m["key"] == "totals":
                    tot += [o["point"] for o in m["outcomes"] if o.get("name") == "Over" and o.get("point")]
        if not tot:
            continue
        total = sum(tot) / len(tot)
        for side in (ev.get("home_team"), ev.get("away_team")):
            ab = by_name.get(str(side).lower())
            if ab and spread.get(side):
                sp = sum(spread[side]) / len(spread[side])      # négatif = favori
                out[ab] = round((total - sp) / 2, 1)
    return out
