"""Entraîne le modèle sur les saisons passées (lancé chaque lundi).

    python -m nhl.train
"""
import json

import pandas as pd

from . import config as C
from . import sources as S
from .features import build
from .model import train


def load_history(seasons):
    sk = pd.concat([S.mp_games("skaters", s) for s in seasons], ignore_index=True)
    gl = pd.concat([S.mp_games("goalies", s) for s in seasons], ignore_index=True)
    tm = S.mp_teams(seasons)
    return sk, gl, tm


def main():
    seasons = C.TRAIN_SEASONS + [C.CURRENT_SEASON]
    print(f"Chargement des saisons {seasons}")
    sk, gl, tm = load_history(seasons)
    print(f"  {len(sk):,} lignes joueurs · {len(gl):,} gardiens · {len(tm):,} équipes")
    df = build(sk, gl, tm)
    metrics, imp = train(df)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print("Poids des paramètres :")
    for x in imp:
        print(f"  {x['param']:<32} {x['share']:>5} %")
    publish_model_page(metrics, imp)


def publish_model_page(metrics, imp):
    """Écrit docs/model.json pour l'écran « Le modèle »."""
    C.DOCS.mkdir(exist_ok=True)
    (C.DOCS / "model.json").write_text(json.dumps({
        "importance": imp, "backtest": metrics,
        "seasons": f"{C.TRAIN_SEASONS[0]}-{C.CURRENT_SEASON + 1}",
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
