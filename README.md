# Buteurs du soir — NHL

App iPhone qui estime, pour chaque joueur des matchs NHL de la nuit, sa probabilité de marquer,
et explique pourquoi à partir de 19 paramètres.

## Fonctionnement
| Heure (Paris) | Tâche | Ce qui se passe |
|---|---|---|
| 8h30 | `results` | vérifie les prédictions de la nuit, met à jour l'écran Fiabilité |
| 9h | `preview` | première estimation des matchs de la nuit (« provisoire ») |
| 18h | `update` | blessures, gardiens probables, cotes |
| minuit | `final` | version finale, archivée dans `history/` puis évaluée |
| lundi 6h | `train` | ré-entraînement du modèle sur les saisons passées |

Tout tourne gratuitement dans GitHub Actions (`.github/workflows/nhl.yml`). L'app (`docs/`) est servie par GitHub Pages.

## Le modèle
- `nhl/features.py` : les 19 paramètres, calculés **uniquement à partir des matchs précédents**
  (le même code sert à l'entraînement et aux prédictions, donc aucune triche possible).
- `nhl/model.py` : LightGBM (arbres boostés), puis calibration pour que « 40 % » veuille dire 40 %.
  Chaque prédiction est décomposée paramètre par paramètre (contributions SHAP), c'est ce qu'affiche la fiche joueur.
- Les cotes bookmakers (total de buts par équipe) ajustent la prédiction à 50 % si la clé est fournie.

## Sources
- API NHL officielle (sans clé) : calendrier, effectifs, gardiens, résultats.
- [MoneyPuck.com](https://moneypuck.com/data.htm) : xG, temps de glace, gardiens, équipes. Usage non commercial, crédit affiché dans l'app.
- [The Odds API](https://the-odds-api.com) (optionnel) : ajouter le secret `ODDS_API_KEY` dans Settings → Secrets → Actions.

## Installation
1. Pousser ce dossier dans un dépôt GitHub.
2. Settings → Pages → branche `main`, dossier `/docs`.
3. Actions → « NHL — prédictions et résultats » → Run workflow → `train` (≈ 10-20 min la première fois).
4. Puis Run workflow → `update`.
5. Sur iPhone : ouvrir l'URL Pages dans Safari → Partager → « Sur l'écran d'accueil ».

## Test hors ligne
`python -m tests.simulate` fait tourner toute la chaîne sur des données simulées.

## Volet NBA : « 20+ du soir »
Le sélecteur 🏒 NHL / 🏀 NBA en haut de l'app bascule vers la probabilité que chaque joueur NBA
marque **20 points ou plus** (filtre 30+ disponible). Lien direct : `…/nhl-buteurs/?sport=nba#bilan`.

- Code : `nba/` (même structure que `nhl/`), tâche `.github/workflows/nba.yml` (toutes les heures à :47, ré-entraînement le lundi).
- Données publiées dans `docs/nba/`, modèle dans `models/nba/`, archives dans `history/nba/`.
- Sources : API publique d'ESPN pour tout (calendrier, feuilles de match, blessures) : stats.nba.com et
  cdn.nba.com refusent les serveurs de GitHub. Le premier entraînement télécharge chaque feuille de match
  (quelques minutes), ensuite tout est en cache. Cotes : même clé `ODDS_API_KEY` (total et écart par équipe).
- Première mise en route : Actions → « NBA — prédictions 20+ et résultats » → Run workflow → `train`, puis `update`.
- Test hors ligne : `python -m tests.simulate_nba`.
