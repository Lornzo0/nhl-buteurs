"""Entraînement, calibration et explication des prédictions.

Modèle : LightGBM (arbres de décision boostés), qui apprend tout seul le poids
de chaque paramètre et leurs interactions (ex. un bon gardien pèse plus face
à un tireur d'élite que face à un joueur de 4e ligne). On le calibre ensuite
(Platt : une petite régression logistique, plus stable que l'isotonique
sur les probabilités élevées où il y a peu d'exemples) pour que « 40 % » corresponde vraiment à 40 % de buts.
"""
import json
import math
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


def train(df):
    d = df[~df["future"] & df["f_games"].ge(5)].dropna(subset=["y"]).sort_values("date")
    cut = d["date"].quantile(0.80)                    # 20 % les plus récents = validation
    tr, va = d[d["date"] < cut], d[d["date"] >= cut]
    print(f"  entraînement : {len(tr):,} matchs-joueurs · validation : {len(va):,}")
    params = {**PARAMS, "monotone_constraints": _monotone()}
    ds_tr = lgb.Dataset(tr[FEATURE_LIST], tr["y"])
    ds_va = lgb.Dataset(va[FEATURE_LIST], va["y"], reference=ds_tr)
    booster = lgb.train(params, ds_tr, 3000, valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    raw = booster.predict(va[FEATURE_LIST], num_iteration=booster.best_iteration)
    iso = Platt().fit(raw, va["y"].to_numpy())
    p = iso.predict(raw)

    # Référence naïve : la moyenne de buts du joueur sur la saison.
    naive = np.clip(1 - np.exp(-va["f_ixg_season"].fillna(0.15).to_numpy() * 1.05), 0.005, 0.9)
    metrics = evaluate(va.assign(p=p), naive)

    # Ré-entraînement final sur tout l'historique, même nombre d'arbres.
    final = lgb.train(params, lgb.Dataset(d[FEATURE_LIST], d["y"]), booster.best_iteration)
    imp = importance(final)
    C.MODELS.mkdir(exist_ok=True)
    final.save_model(str(C.MODELS / "lgbm.txt"))
    with open(C.MODELS / "calibration.pkl", "wb") as f:
        pickle.dump(iso, f)
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (C.MODELS / "importance.json").write_text(json.dumps(imp, ensure_ascii=False, indent=1))
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
