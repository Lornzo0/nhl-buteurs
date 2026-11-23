"""Entraînement, calibration et explication des prédictions.

Modèle : LightGBM (arbres de décision boostés), qui apprend tout seul le poids
de chaque paramètre et leurs interactions (ex. un bon gardien pèse plus face
à un tireur d'élite que face à un joueur de 4e ligne). On le calibre ensuite
(Platt : une petite régression logistique, plus stable que l'isotonique
sur les probabilités élevées où il y a peu d'exemples) pour que « 40 % » corresponde vraiment à 40 % de buts.
"""
import json
import math
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config as C
from .features import FEATURE_LIST, FEATURES, MONOTONE

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=7)
# Paramètres de base qui forment la « base du joueur » dans l'explication.
BASE_PARAMS = {"Volume de tirs", "Qualité des tirs", "Niveau habituel"}


class Platt:
    """Calibration : p_calibrée = sigmoïde(a · logit(p_brute) + b)."""
    def fit(self, raw, y):
        self.lr = LogisticRegression(C=1e6).fit(_lg(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(_lg(np.asarray(raw)).reshape(-1, 1))[:, 1]


def _lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _monotone():
    return [MONOTONE.get(f, 0) for f in FEATURE_LIST]


# Variantes testées chaque lundi : on garde celle qui prédit le mieux.
VARIANTS = [
    {"num_leaves": 15, "min_child_samples": 500},
    {"num_leaves": 31, "min_child_samples": 300},
    {"num_leaves": 63, "min_child_samples": 200, "learning_rate": 0.02},
]
HALF_LIFE_DAYS = 365          # un match d'il y a 1 an compte 2 fois moins qu'hier
CHAMPION_DAYS = 28            # fenêtre de comparaison avec le modèle en place
MIN_DUEL = int(os.environ.get("NHL_MIN_DUEL", 2000))   # prédictions minimum pour juger


def recency(dates, ref):
    return 0.5 ** ((ref - dates).dt.days.to_numpy() / HALF_LIFE_DAYS)


def _fit(params, X, y, w, rounds, valid=None):
    ds = lgb.Dataset(X, y, weight=w)
    if valid is None:
        return lgb.train(params, ds, rounds)
    dv = lgb.Dataset(valid[0], valid[1], reference=ds)
    return lgb.train(params, ds, rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def champion_rows():
    """Prédictions réelles (faites AVANT les matchs) du modèle en place, vérifiées."""
    rows = []
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("graded"):
            continue
        for pid, c in d["players"].items():
            if c.get("scored") is not None:
                rows.append({"date": pd.Timestamp(d["date"]), "player_id": int(pid),
                             "p_champ": c["prob"], "y_live": float(c["scored"] > 0)})
    return pd.DataFrame(rows)


def train(df):
    d = df[~df["future"] & df["f_games"].ge(5)].dropna(subset=["y"]).sort_values("date")
    ref = d["date"].max()
    cut = d["date"].quantile(0.80)                    # 20 % les plus récents = validation
    tr, va = d[d["date"] < cut], d[d["date"] >= cut]
    print(f"  entraînement : {len(tr):,} matchs-joueurs · validation : {len(va):,}")

    # 1. Tester les variantes, garder la meilleure sur la validation.
    best = None
    for v in VARIANTS:
        params = {**PARAMS, **v, "monotone_constraints": _monotone()}
        b = _fit(params, tr[FEATURE_LIST], tr["y"], recency(tr["date"], cut), 3000,
                 (va[FEATURE_LIST], va["y"]))
        raw = b.predict(va[FEATURE_LIST], num_iteration=b.best_iteration)
        ll = log_loss(va["y"], np.clip(raw, 1e-4, 1 - 1e-4))
        print(f"  variante {v} → {ll:.4f} ({b.best_iteration} arbres)")
        if best is None or ll < best[0]:
            best = (ll, params, b.best_iteration, raw, v)
    _, params, n_iter, raw, variant = best

    # 2. Calibration sur les matchs les plus récents.
    cal = Platt().fit(raw, va["y"].to_numpy())
    p = cal.predict(raw)
    naive = np.clip(1 - np.exp(-va["f_ixg_season"].fillna(0.15).to_numpy() * 1.05), 0.005, 0.9)
    metrics = evaluate(va.assign(p=p), naive)
    metrics["variant"] = variant

    # 3. Duel avec le modèle en place, sur ses vraies prédictions des 4 dernières semaines.
    champ = champion_rows()
    duel = None
    if len(champ):
        win_start = champ["date"].max() - pd.Timedelta(days=CHAMPION_DAYS)
        champ = champ[champ["date"] > win_start]
        before = d[d["date"] <= win_start]
        chal = _fit(params, before[FEATURE_LIST], before["y"], recency(before["date"], win_start), n_iter)
        win = d[d["date"] > win_start].merge(champ, on=["date", "player_id"])
        if len(win) >= MIN_DUEL:
            p_chal = cal.predict(chal.predict(win[FEATURE_LIST]))
            ll_chal, ll_champ = log_loss(win["y"], p_chal), log_loss(win["y"], np.clip(win["p_champ"], 1e-4, 1 - 1e-4))
            duel = {"n": int(len(win)), "challenger": round(ll_chal, 4), "champion": round(ll_champ, 4),
                    "winner": "challenger" if ll_chal <= ll_champ else "champion"}
            print(f"  duel sur {len(win):,} prédictions réelles : nouveau {ll_chal:.4f} vs en place {ll_champ:.4f}")
    metrics["duel"] = duel
    feats_file = C.MODELS / "features.json"
    same_feats = feats_file.exists() and json.loads(feats_file.read_text()) == FEATURE_LIST
    if duel and duel["winner"] == "champion" and same_feats:
        print("  → le modèle en place reste meilleur : on le garde.")
        old = json.loads((C.MODELS / "metrics.json").read_text())
        old["duel"] = duel
        (C.MODELS / "metrics.json").write_text(json.dumps(old, ensure_ascii=False, indent=1))
        return old, json.loads((C.MODELS / "importance.json").read_text())

    # 4. Ré-entraînement final sur tout l'historique, matchs récents favorisés.
    final = _fit(params, d[FEATURE_LIST], d["y"], recency(d["date"], ref), n_iter)
    imp = importance(final)
    C.MODELS.mkdir(exist_ok=True)
    final.save_model(str(C.MODELS / "lgbm.txt"))
    with open(C.MODELS / "calibration.pkl", "wb") as f:
        pickle.dump(cal, f)
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (C.MODELS / "importance.json").write_text(json.dumps(imp, ensure_ascii=False, indent=1))
    feats_file.write_text(json.dumps(FEATURE_LIST))
    return metrics, imp


def evaluate(va, naive):
    y, p = va["y"].to_numpy(), va["p"].to_numpy()
    bins = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 1.0)]
    cal = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            cal.append({"bin": f"{round(lo*100)}–{round(hi*100)} %" if hi < 1 else f"{round(lo*100)} %+",
                        "pred": round(float(p[m].mean()), 3), "obs": round(float(y[m].mean()), 3),
                        "n": int(m.sum())})
    # Top 10 de chaque soirée : combien ont marqué ?
    def top10(score):
        t = va.assign(s=score).sort_values("s", ascending=False)
        t = t.groupby(t["date"].dt.date).head(10)
        return t.groupby(t["date"].dt.date)["y"].sum()
    hits, hits_naive = top10(p), top10(naive)
    ll, ll_naive = log_loss(y, p), log_loss(y, naive)
    return {
        "n": int(len(y)), "nights": int(hits.size),
        "logloss": round(ll, 4), "logloss_naive": round(ll_naive, 4),
        "top10_naive": round(float(hits_naive.mean()), 2),
        "gain_vs_naive": round((hits.mean() - hits_naive.mean()) / hits_naive.mean() * 100, 1),
        "auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 4),
        "top10_hits": round(float(hits.mean()), 2), "calibration": cal,
        "period": [str(va["date"].min().date()), str(va["date"].max().date())],
    }


def importance(booster):
    gain = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
    agg = {}
    for f, v in gain.items():
        grp, label = FEATURES[f]
        agg.setdefault(label, {"group": grp, "gain": 0.0})["gain"] += float(v)
    total = sum(x["gain"] for x in agg.values()) or 1
    return sorted([{"param": k, "group": v["group"], "share": round(v["gain"] / total * 100, 1)}
                   for k, v in agg.items()], key=lambda x: -x["share"])


def load():
    booster = lgb.Booster(model_file=str(C.MODELS / "lgbm.txt"))
    with open(C.MODELS / "calibration.pkl", "rb") as f:
        iso = pickle.load(f)
    return booster, iso


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(booster, iso, X):
    """Probabilités calibrées + contributions de chaque paramètre."""
    raw = booster.predict(X[FEATURE_LIST])
    contrib = booster.predict(X[FEATURE_LIST], pred_contrib=True)   # log-odds, + biais en dernière colonne
    cal = iso.predict(raw)
    out = []
    for i in range(len(X)):
        c = contrib[i]
        by_param = {}
        for f, v in zip(FEATURE_LIST, c[:-1]):
            grp, label = FEATURES[f]
            by_param.setdefault((grp, label), 0.0)
            by_param[(grp, label)] += float(v)
        # La calibration décale un peu la note : on répartit l'écart sur la base.
        shift = logit(cal[i]) - logit(raw[i])
        base_logit = c[-1] + shift + sum(v for (g, l), v in by_param.items() if l in BASE_PARAMS)
        factors = [{"group": g, "param": l, "mult": round(math.exp(v), 3)}
                   for (g, l), v in by_param.items() if l not in BASE_PARAMS]
        out.append({"p": float(cal[i]), "base": sigmoid(base_logit), "factors": factors})
    return out
