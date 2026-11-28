"""Sources de données NBA (tout vient d'ESPN, joignable depuis GitHub Actions).

stats.nba.com et cdn.nba.com refusent les serveurs de GitHub : on utilise
donc l'API publique d'ESPN pour tout, ce qui garde des identifiants de
joueurs cohérents entre l'historique, les prédictions et les résultats.

  - scoreboard?dates=AAAAMMJJ : matchs d'une soirée (date de New York)
  - summary?event=ID : feuille de match joueur par joueur
  - injuries : rapport de blessures (joueurs « Out » retirés)
  - The Odds API (optionnel, ODDS_API_KEY) : total de points et écart par équipe
"""
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import config as C

S = requests.Session()
S.headers["User-Agent"] = "nhl-buteurs/2.0 (projet perso non commercial)"
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
NY = ZoneInfo("America/New_York")
LOG_COLS = ["player_id", "name", "team", "opp", "home", "game_id", "date", "season", "min", "pts",
            "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "reb", "ast", "tov", "starter"]
GAME_TYPES = {2, 3, 5}          # saison régulière, séries, play-in
WORKERS = 8


def http(url, params=None, retries=3, timeout=30):
    for i in range(retries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == retries - 1:
                print(f"! échec {url} {params or ''}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))


def fix(t):
    return C.TEAM_FIX.get(t, t)


def season_str(s):
    return f"{s}-{str(s + 1)[2:]}"


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def _cache_dir(name):
    p = C.CACHE / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- calendrier

def scoreboard(day, cache=False):
    """Matchs ESPN d'une soirée (liste brute d'événements)."""
    f = _cache_dir("scoreboard") / f"{day:%Y%m%d}.json"
    if cache and f.exists():
        return json.loads(f.read_text())
    data = http(C.ESPN_SCOREBOARD, params={"dates": f"{day:%Y%m%d}", "limit": 50})
    if data is None:
        return None
    events = data.get("events", [])
    done = all(((ev.get("status") or {}).get("type") or {}).get("completed")
               or ((ev.get("status") or {}).get("type") or {}).get("state") == "post" for ev in events)
    if cache and done:                   # on ne fige une soirée qu'une fois tous ses matchs finis
        f.write_text(json.dumps(events))
    return events


def _game(ev, day):
    comp = (ev.get("competitions") or [{}])[0]
    stype = (ev.get("season") or {}).get("type") or (comp.get("type") or {}).get("id")
    try:
        stype = int(stype)
    except (TypeError, ValueError):
        stype = 2
    if stype not in GAME_TYPES:
        return None
    teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if "home" not in teams or "away" not in teams:
        return None
    h, a = teams["home"]["team"], teams["away"]["team"]
    st = (comp.get("status") or ev.get("status") or {}).get("type", {})
    return {"game_id": str(ev["id"]), "day": day.isoformat(), "start": ev.get("date"),
            "state": st.get("state"), "final": bool(st.get("completed")),
            "home": fix(h.get("abbreviation")), "away": fix(a.get("abbreviation")),
            "home_name": h.get("shortDisplayName") or h.get("name") or h.get("abbreviation"),
            "away_name": a.get("shortDisplayName") or a.get("name") or a.get("abbreviation"),
            "venue": (comp.get("venue") or {}).get("fullName", "")}


def schedule(day):
    """Matchs d'une soirée NBA (date à l'heure de New York)."""
    return [g for g in (_game(ev, day) for ev in (scoreboard(day) or [])) if g]


# ---------------------------------------------------------------- feuilles de match

def _split(v):
    m = re.match(r"(\d+)-(\d+)", str(v))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _num(v):
    try:
        return float(str(v).replace("+", ""))
    except ValueError:
        return 0.0


def box(gid):
    """Feuille de match ESPN : (terminé ?, lignes joueurs) ; (None, []) si injoignable."""
    data = http(C.ESPN_SUMMARY, params={"event": gid})
    if not data:
        return None, []
    comp = ((data.get("header") or {}).get("competitions") or [{}])[0]
    final = bool(((comp.get("status") or {}).get("type") or {}).get("completed"))
    side = {c.get("team", {}).get("abbreviation"): c.get("homeAway") for c in comp.get("competitors", [])}
    blocks = (data.get("boxscore") or {}).get("players", [])
    abbrs = [b.get("team", {}).get("abbreviation") for b in blocks]
    rows = []
    for b in blocks:
        team = b.get("team", {}).get("abbreviation")
        opp = next((x for x in abbrs if x != team), None)
        for st in b.get("statistics", []):
            labels = [str(x).upper() for x in (st.get("labels") or st.get("names") or [])]
            for a in st.get("athletes", []):
                vals = a.get("stats") or []
                if a.get("didNotPlay") or len(vals) != len(labels):
                    continue
                s = dict(zip(labels, vals))
                mins = _num(s.get("MIN", 0))
                if mins <= 0:
                    continue
                fgm, fga = _split(s.get("FG"))
                f3m, f3a = _split(s.get("3PT"))
                ftm, fta = _split(s.get("FT"))
                ath = a.get("athlete") or {}
                rows.append({
                    "player_id": int(ath["id"]), "name": ath.get("displayName", ""),
                    "team": fix(team), "opp": fix(opp), "home": int(side.get(team) == "home"),
                    "game_id": str(gid), "min": mins, "pts": int(_num(s.get("PTS", 0))),
                    "fga": fga, "fgm": fgm, "fg3a": f3a, "fg3m": f3m, "fta": fta, "ftm": ftm,
                    "oreb": int(_num(s.get("OREB", 0))), "reb": int(_num(s.get("REB", 0))),
                    "ast": int(_num(s.get("AST", 0))), "tov": int(_num(s.get("TO", 0))),
                    "starter": 1.0 if a.get("starter") else 0.0})
    return final, rows


def boxscore_points(gid):
    """{player_id: points} pour un match terminé, ou None s'il n'est pas fini / introuvable."""
    final, rows = box(gid)
    if not final:
        return None
    return {r["player_id"]: r["pts"] for r in rows}


def game_logs(season):
    """Une ligne par joueur et par match joué d'une saison (saison régulière + play-in + séries).

    Chaque soirée et chaque feuille de match est mise en cache : la première
    saison prend quelques minutes, les passages suivants ne téléchargent que
    les nouveaux matchs."""
    p = C.CACHE / f"logs_{season}.csv.gz"
    max_age = 5 if season == C.CURRENT_SEASON else 24 * 30
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age * 3600:
        return _typed(pd.read_csv(p, dtype={"game_id": str}))
    today = datetime.now(NY).date()
    day, end = date(season, 10, 1), min(date(season + 1, 6, 30), today - timedelta(days=1))
    days = []
    while day <= end:
        days.append(day)
        day += timedelta(days=1)
    print(f"  ESPN : saison {season_str(season)}, {len(days)} jours")
    with ThreadPoolExecutor(WORKERS) as ex:
        boards = list(ex.map(lambda d: (d, scoreboard(d, cache=True)), days))
    games = []
    fails = sum(1 for _, ev in boards if ev is None)
    if fails > len(days) // 2:
        print(f"  ! ESPN ne répond pas ({fails} soirées sur {len(days)} en échec)", file=sys.stderr)
    for d, evs in boards:
        for ev in evs or []:
            g = _game(ev, d)
            if g and g["final"]:
                games.append(g)
    bdir = _cache_dir(f"box_{season}")

    def one(g):
        f = bdir / f"{g['game_id']}.json"
        if f.exists():
            rows = json.loads(f.read_text())
        else:
            final, rows = box(g["game_id"])
            if not final:
                return []
            f.write_text(json.dumps(rows))
        for r in rows:
            r["date"] = g["day"]
        return rows
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = [r for rs in ex.map(one, games) for r in rs]
    print(f"    {len(games)} matchs, {len(rows):,} lignes joueurs")
    d = pd.DataFrame(rows, columns=[c for c in LOG_COLS if c != "season"])
    if d.empty:
        if p.exists():
            print("  (on garde la copie en cache)")
            return _typed(pd.read_csv(p, dtype={"game_id": str}))
        return pd.DataFrame(columns=LOG_COLS)
    d["season"] = season
    d.to_csv(p, index=False)
    return _typed(d)


def _typed(d):
    d = d.copy()
    d["game_id"] = d["game_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    for c in ("team", "opp"):
        d[c] = d[c].map(fix)
    return d[LOG_COLS]


def team_logs(pl):
    """Totaux d'équipe par match, calculés à partir des lignes joueurs."""
    t = (pl.groupby(["game_id", "date", "season", "team", "opp", "home"], as_index=False)
         [["pts", "fga", "fta", "oreb", "tov"]].sum())
    t["poss"] = t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    o = t[["game_id", "team", "pts", "poss"]].rename(columns={"team": "opp", "pts": "opp_pts", "poss": "opp_poss"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["poss"] = (t["poss"] + t["opp_poss"]) / 2          # possessions du match (moyenne des deux équipes)
    return t.drop(columns="opp_poss")


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
