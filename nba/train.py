"""Entraîne le modèle NBA sur les saisons passées (lancé chaque lundi).

    python -m nba.train
"""
import json

import pandas as pd

from . import config as C
from . import sources as S
from .features import build
from .model import train


def load_history(seasons):
    frames = [S.game_logs(s) for s in seasons]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("Aucune donnée NBA téléchargée (ESPN ne répond pas).")
    return pd.concat(frames, ignore_index=True)


def main():
    seasons = C.TRAIN_SEASONS + [C.CURRENT_SEASON]
    print(f"Chargement des saisons NBA {seasons}")
    pl = load_history(seasons)
    print(f"  {len(pl):,} lignes joueurs · {pl['game_id'].nunique():,} matchs")
    df = build(pl)
    metrics, imp = train(df)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print("Poids des paramètres :")
    for x in imp:
        print(f"  {x['param']:<36} {x['share']:>5} %")
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "model.json").write_text(json.dumps({
        "importance": imp, "backtest": metrics,
        "seasons": f"{C.TRAIN_SEASONS[0]}-{C.CURRENT_SEASON + 1}",
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
