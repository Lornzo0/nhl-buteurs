"""Constantes du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"            # téléchargements (non versionnés)
MODELS = ROOT / "models"           # modèle entraîné (versionné)
HISTORY = ROOT / "history"         # prédictions finales archivées (versionné)
DOCS = ROOT / "docs"               # l'app (GitHub Pages)

NHL_WEB = "https://api-web.nhle.com/v1"
MP_GAMES = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/{kind}/{season}.zip"
MP_TEAMS_ALL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"

# Saison MoneyPuck = année de début (2026 = saison 2026-27).
CURRENT_SEASON = 2026
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Abréviations : MoneyPuck utilise parfois des points (T.B, N.J, L.A, S.J).
TEAM_FIX = {"T.B": "TBL", "N.J": "NJD", "L.A": "LAK", "S.J": "SJS",
            "ARI": "UTA", "PHX": "UTA"}

# Coordonnées des arénas (lat, lon) pour la distance de déplacement.
ARENAS = {
    "ANA": (33.808, -117.877), "BOS": (42.366, -71.062), "BUF": (42.875, -78.876),
    "CAR": (35.803, -78.722), "CBJ": (39.969, -83.006), "CGY": (51.037, -114.052),
    "CHI": (41.881, -87.674), "COL": (39.749, -105.008), "DAL": (32.790, -96.810),
    "DET": (42.341, -83.055), "EDM": (53.547, -113.498), "FLA": (26.158, -80.326),
    "LAK": (34.043, -118.267), "MIN": (44.945, -93.101), "MTL": (45.496, -73.569),
    "NJD": (40.734, -74.171), "NSH": (36.159, -86.779), "NYI": (40.712, -73.726),
    "NYR": (40.751, -73.993), "OTT": (45.297, -75.927), "PHI": (39.901, -75.172),
    "PIT": (40.439, -79.989), "SEA": (47.622, -122.354), "SJS": (37.333, -121.901),
    "STL": (38.627, -90.203), "TBL": (27.943, -82.452), "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901), "VAN": (49.278, -123.109), "VGK": (36.103, -115.178),
    "WPG": (49.893, -97.144), "WSH": (38.898, -77.021),
}

# Familles de paramètres affichées dans l'app.
GROUPS = {
    "player": "Le joueur", "opp": "L'adversaire",
    "ctx": "Le contexte", "book": "Les bookmakers",
}
