"""Entraînement, calibration et explication du modèle NBA.

Mêmes briques que la NHL (LightGBM + calibration de Platt + duel avec le
modèle en place), appliquées à la cible « 20 points ou plus ».
"""
import json
import math
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nhl.model import PARAMS, VARIANTS, Platt, _fit, logit, recency, sigmoid
from . import config as C
from .features import FEATURE_LIST, FEATURES, MONOTONE

BASE_PARAMS = {"Niveau habituel", "Volume de tirs", "Rôle offensif"}
CHAMPION_DAYS = 28
MIN_DUEL = int(os.environ.get("NBA_MIN_DUEL", 1500))
CUT = C.THRESHOLD - 0.5


def _monotone():
    return [MONOTONE.get(f, 0) for f in FEATURE_LIST]


def naive_prob(d):
    """Référence simple : moyenne de points de la saison et dispersion habituelle (loi normale)."""
    mu = d["f_pts_season"].fillna(d["f_pts_ewm"]).fillna(8).to_numpy()
    sd = np.maximum(4, d["f_pts_sd"].fillna(6).to_numpy())
    return np.clip(1 - norm.cdf((CUT - mu) / sd), 0.002, 0.98)


def champion_rows():
    rows = []
    for f in sorted(C.HISTORY.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("graded"):
            continue
        for pid, c in d["players"].items():
            if c.get("pts_real") is not None:
                rows.append({"date": pd.Timestamp(d["date"]), "player_id": int(pid),
                             "p_champ": c["prob"], "y_live": float(c["pts_real"] >= C.THRESHOLD)})
    return pd.DataFrame(rows)


def train(df):
    d = df[~df["future"] & df["f_games"].ge(5)].dropna(subset=["y"]).sort_values("date")
    ref = d["date"].max()
    cut = d["date"].quantile(0.80)
    tr, va = d[d["date"] < cut], d[d["date"] >= cut]
    print(f"  entraînement : {len(tr):,} matchs-joueurs · validation : {len(va):,}")

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

    cal = Platt().fit(raw, va["y"].to_numpy())
    p = cal.predict(raw)
    metrics = evaluate(va.assign(p=p), naive_prob(va))
    metrics["variant"] = variant

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

    final = _fit(params, d[FEATURE_LIST], d["y"], recency(d["date"], ref), n_iter)
    imp = importance(final)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    final.save_model(str(C.MODELS / "lgbm.txt"))
    with open(C.MODELS / "calibration.pkl", "wb") as f:
        pickle.dump(cal, f)
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (C.MODELS / "importance.json").write_text(json.dumps(imp, ensure_ascii=False, indent=1))
    feats_file.write_text(json.dumps(FEATURE_LIST))
    return metrics, imp


def evaluate(va, naive):
    y, p = va["y"].to_numpy(), va["p"].to_numpy()
    bins = [(0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]
    cal = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            cal.append({"bin": f"{round(lo*100)}–{round(hi*100)} %" if hi < 1 else f"{round(lo*100)} %+",
                        "pred": round(float(p[m].mean()), 3), "obs": round(float(y[m].mean()), 3),
                        "n": int(m.sum())})

    def top10(score):
        t = va.assign(s=score).sort_values("s", ascending=False)
        t = t.groupby(t["date"].dt.date).head(10)
        return t.groupby(t["date"].dt.date)["y"].sum()
    hits, hits_naive = top10(p), top10(naive)
    return {
        "n": int(len(y)), "nights": int(hits.size),
        "logloss": round(log_loss(y, p), 4), "logloss_naive": round(log_loss(y, naive), 4),
        "top10_naive": round(float(hits_naive.mean()), 2),
        "gain_vs_naive": round((hits.mean() - hits_naive.mean()) / max(hits_naive.mean(), 1e-9) * 100, 1),
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
        cal = pickle.load(f)
    return booster, cal


def predict(booster, cal, X):
    raw = booster.predict(X[FEATURE_LIST])
    contrib = booster.predict(X[FEATURE_LIST], pred_contrib=True)
    p = cal.predict(raw)
    out = []
    for i in range(len(X)):
        c = contrib[i]
        by_param = {}
        for f, v in zip(FEATURE_LIST, c[:-1]):
            key = FEATURES[f]
            by_param[key] = by_param.get(key, 0.0) + float(v)
        shift = logit(p[i]) - logit(raw[i])
        base_logit = c[-1] + shift + sum(v for (g, l), v in by_param.items() if l in BASE_PARAMS)
        factors = [{"group": g, "param": l, "mult": round(math.exp(v), 3)}
                   for (g, l), v in by_param.items() if l not in BASE_PARAMS]
        out.append({"p": float(p[i]), "base": sigmoid(base_logit), "factors": factors})
    return out
