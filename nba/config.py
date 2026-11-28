"""Constantes du volet NBA (probabilité de marquer 20 points ou plus)."""
from nhl.config import ROOT

CACHE = ROOT / ".cache" / "nba"          # téléchargements (non versionnés)
MODELS = ROOT / "models" / "nba"          # modèle entraîné (versionné)
HISTORY = ROOT / "history" / "nba"        # prédictions finales archivées (versionné)
DOCS = ROOT / "docs" / "nba"              # données de l'app (GitHub Pages)

THRESHOLD = 20                            # le « 20+ » de l'app

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_SCOREBOARD = ESPN + "/scoreboard"
ESPN_SUMMARY = ESPN + "/summary"
ESPN_INJURIES = ESPN + "/injuries"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Saison = année de début (2026 = saison 2026-27).
CURRENT_SEASON = 2026
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Codes historiques → codes actuels.
TEAM_FIX = {"NOH": "NOP", "NJN": "BKN", "SEA": "OKC", "PHO": "PHX", "GS": "GSW", "SA": "SAS",
            "NY": "NYK", "NO": "NOP", "WSH": "WAS", "UTAH": "UTA"}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
    "IND": "Indiana Pacers", "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Coordonnées des salles (lat, lon) pour la distance de déplacement.
ARENAS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.497, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (33.945, -118.343), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188), "MIL": (43.045, -87.917), "MIN": (44.980, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.993), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}
