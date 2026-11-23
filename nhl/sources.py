"""Téléchargement et normalisation des sources de données.

Trois sources :
  - MoneyPuck (données match par match : xG, temps de glace, gardiens, équipes)
  - API NHL officielle (calendrier, effectifs, résultats)
  - The Odds API (cotes, optionnel : variable d'environnement ODDS_API_KEY)

Les noms de colonnes MoneyPuck sont résolus via des listes de candidats :
si MoneyPuck renomme une colonne, le script s'arrête avec un message clair
plutôt que de produire des prédictions fausses en silence.
"""
import io
import os
import sys
import time
import zipfile
from datetime import date, datetime

import pandas as pd
import requests

from . import config as C

S = requests.Session()
S.headers["User-Agent"] = "nhl-buteurs/2.0 (projet perso non commercial)"


def http(url, params=None, retries=3, binary=False):
    for i in range(retries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binary else r.json()
        except requests.RequestException as e:
            if i == retries - 1:
                print(f"! échec {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------- MoneyPuck

SCHEMA = {
    "skaters": {
        "player_id": ["playerId"], "name": ["name"], "game_id": ["gameId"],
        "date": ["gameDate"], "team": ["playerTeam", "team"],
        "opp": ["opposingTeam"], "venue": ["home_or_away"],
        "pos": ["position"], "situation": ["situation"], "toi": ["icetime", "iceTime"],
        "goals": ["I_F_goals"], "ixg": ["I_F_xGoals"], "sog": ["I_F_shotsOnGoal"],
        "fenwick": ["I_F_unblockedShotAttempts"], "hd": ["I_F_highDangerShots"],
        "md": ["I_F_mediumDangerShots"], "ld": ["I_F_lowDangerShots"],
        "onice_xgf": ["OnIce_F_xGoals"],
        "points": ["I_F_points", "?"],          # "?" = colonne facultative
    },
    "goalies": {
        "player_id": ["playerId"], "name": ["name"], "game_id": ["gameId"],
        "date": ["gameDate"], "team": ["playerTeam", "team"], "opp": ["opposingTeam"],
        "situation": ["situation"], "toi": ["icetime", "iceTime"],
        "xga": ["xGoals"], "ga": ["goals"],
    },
    "teams": {
        "game_id": ["gameId"], "date": ["gameDate"], "team": ["playerTeam", "team"],
        "opp": ["opposingTeam"], "venue": ["home_or_away"], "season": ["season"],
        "situation": ["situation"], "toi": ["iceTime", "icetime"],
        "xgf": ["xGoalsFor"], "xga": ["xGoalsAgainst"], "gf": ["goalsFor"], "ga": ["goalsAgainst"],
        "pen": ["penaltiesFor"], "blocked_against": ["blockedShotAttemptsAgainst"],
    },
}


def normalize(df, kind):
    out, missing = {}, []
    for key, cands in SCHEMA[kind].items():
        col = next((c for c in cands if c in df.columns), None)
        if col is None and "?" in cands:
            out[key] = pd.Series(float("nan"), index=df.index)
            continue
        if col is None:
            missing.append(f"{key} ({' / '.join(cands)})")
        else:
            out[key] = df[col]
    if missing:
        sys.exit(f"Colonnes MoneyPuck introuvables dans '{kind}': {', '.join(missing)}\n"
                 f"Colonnes disponibles : {list(df.columns)[:80]}")
    d = pd.DataFrame(out)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    for c in ("team", "opp"):
        if c in d:
            d[c] = d[c].replace(C.TEAM_FIX)
    if "venue" in d:
        d["home"] = (d["venue"].astype(str).str.upper().str[0] == "H").astype(int)
        d = d.drop(columns="venue")
    return d


def _read_zip_csv(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        frames = [pd.read_csv(z.open(n), low_memory=False)
                  for n in z.namelist() if n.endswith(".csv")]
    return pd.concat(frames, ignore_index=True)


def _cached(name, url, max_age_h):
    C.CACHE.mkdir(exist_ok=True)
    p = C.CACHE / name
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600:
        return p.read_bytes()
    print(f"  téléchargement {url}")
    blob = http(url, binary=True)
    if blob is None:
        if p.exists():
            print("  (on garde la copie en cache)")
            return p.read_bytes()
        return None
    p.write_bytes(blob)
    return blob


def mp_games(kind, season):
    """Données match par match d'une saison (kind = skaters | goalies)."""
    age = 6 if season == C.CURRENT_SEASON else 24 * 30
    blob = _cached(f"mp_{kind}_{season}.zip", C.MP_GAMES.format(kind=kind, season=season), age)
    if blob is None:
        return pd.DataFrame()
    d = normalize(_read_zip_csv(blob), kind)
    d["season"] = season
    return d


def mp_teams(seasons):
    blob = _cached("mp_all_teams.csv", C.MP_TEAMS_ALL, 6)
    if blob is None:
        sys.exit("Impossible de télécharger les données équipes MoneyPuck.")
    raw = pd.read_csv(io.BytesIO(blob), low_memory=False)
    if "season" in raw:
        raw = raw[raw["season"].isin(seasons)]
    d = normalize(raw, "teams")
    return d[d["situation"] == "all"].drop(columns="situation")


# ---------------------------------------------------------------- API NHL

def nhl(path):
    data = http(C.NHL_WEB + path)
    time.sleep(0.1)
    return data


def _name(x):
    return (x.get("fr") or x.get("default", "")) if isinstance(x, dict) else (x or "")


def schedule(day: date):
    """Matchs d'une soirée NHL (date à l'heure de New York)."""
    data = nhl(f"/schedule/{day.isoformat()}") or {}
    games = []
    for d in data.get("gameWeek", []):
        if d.get("date") != day.isoformat():
            continue
        for g in d.get("games", []):
            if g.get("gameType") not in (2, 3, None):   # 2 = saison, 3 = séries
                continue
            a, h = g["awayTeam"], g["homeTeam"]
            games.append({
                "game_id": g["id"], "start": g.get("startTimeUTC"), "state": g.get("gameState"),
                "away": C.TEAM_FIX.get(a["abbrev"], a["abbrev"]), "home": C.TEAM_FIX.get(h["abbrev"], h["abbrev"]),
                "away_name": _name(a.get("commonName")) or a["abbrev"],
                "home_name": _name(h.get("commonName")) or h["abbrev"],
                "venue": _name(g.get("venue")),
            })
    return games


def roster(team):
    """Joueurs actuellement sous contrat avec l'équipe (id → nom, poste)."""
    data = nhl(f"/roster/{team}/current") or {}
    out = {}
    for grp in ("forwards", "defensemen", "goalies"):
        for p in data.get(grp, []):
            out[p["id"]] = {"name": f"{_name(p.get('firstName'))} {_name(p.get('lastName'))}".strip(),
                            "pos": p.get("positionCode", "")}
    return out


def boxscore_goals(game_id):
    """{playerId: buts} pour un match terminé, ou None s'il n'est pas fini."""
    data = nhl(f"/gamecenter/{game_id}/boxscore")
    if not data or data.get("gameState") not in ("OFF", "FINAL"):
        return None
    goals = {}
    stats = data.get("playerByGameStats", {})
    for side in ("awayTeam", "homeTeam"):
        for grp in ("forwards", "defense"):
            for p in stats.get(side, {}).get(grp, []):
                goals[p["playerId"]] = p.get("goals", 0) or 0
    return goals


def goalie_starts(team, n=12):
    """Derniers départs des gardiens d'une équipe, via les feuilles de match NHL."""
    data = nhl(f"/club-schedule-season/{team}/now") or {}
    done = [g for g in data.get("games", []) if g.get("gameState") in ("OFF", "FINAL")][-n:]
    starts = []
    for g in done:
        box = nhl(f"/gamecenter/{g['id']}/boxscore") or {}
        side = "homeTeam" if C.TEAM_FIX.get(g["homeTeam"]["abbrev"], g["homeTeam"]["abbrev"]) == team else "awayTeam"
        gks = box.get("playerByGameStats", {}).get(side, {}).get("goalies", [])
        starter = next((x for x in gks if x.get("starter")), None) or (max(gks, key=lambda x: x.get("toi", "0:00")) if gks else None)
        if starter:
            starts.append({"date": g.get("gameDate"), "id": starter["playerId"],
                           "name": _name(starter.get("name"))})
    return starts


# ---------------------------------------------------------------- Cotes

def team_totals():
    """Buts attendus par équipe selon les bookmakers : {abbrev: buts}. {} sans clé."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    data = http(C.ODDS_URL, params={"apiKey": key, "regions": "us,eu", "markets": "h2h,totals",
                                    "oddsFormat": "decimal"})
    if not data:
        return {}
    # Appariement par surnom ("Canadiens", "Maple Leafs"…) : robuste aux accents.
    def abbrev(full):
        for k, v in TEAM_NAMES.items():
            if full and full.split()[-1].lower() == v.split()[-1].lower():
                return k
        return None
    out = {}
    for ev in data:
        h2h, tot = {}, []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "h2h":
                    for o in m["outcomes"]:
                        h2h.setdefault(o["name"], []).append(1 / o["price"])
                elif m["key"] == "totals":
                    tot += [o.get("point") for o in m["outcomes"] if o.get("name") == "Over" and o.get("point")]
        home, away = ev.get("home_team"), ev.get("away_team")
        if not (tot and home in h2h and away in h2h):
            continue
        total = sum(tot) / len(tot)
        ph, pa = sum(h2h[home]) / len(h2h[home]), sum(h2h[away]) / len(h2h[away])
        p = ph / (ph + pa)                       # proba de victoire domicile, marge retirée
        share = 0.5 + (p - 0.5) * 0.55            # part des buts : ~0,55 point de part par point de proba
        if abbrev(home):
            out[abbrev(home)] = round(total * share, 2)
        if abbrev(away):
            out[abbrev(away)] = round(total * (1 - share), 2)
    return out


TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets", "CGY": "Calgary Flames", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montréal Canadiens",
    "NJD": "New Jersey Devils", "NSH": "Nashville Predators", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken", "SJS": "San Jose Sharks", "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}
