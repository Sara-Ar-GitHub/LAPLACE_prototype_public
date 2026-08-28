# LAPLACE — prototype jouet

**Un scientifique artificiel pour les systèmes physiques en réseau.**

Démonstration de faisabilité de la boucle complète de découverte scientifique :

```
world model → incertitude → curiosité (sous contrainte de sécurité)
            → expérience → révision → loi symbolique → vérification
```

Le système étudié est un réseau de transport à **118 nœuds** simulé avec `pandapower`.
Point décisif : les équations qui gouvernent ce simulateur — écoulement de puissance,
matrice d'admittance `Ybus` — sont **connues par construction**. Toute loi que l'agent
prétend avoir découverte peut donc être **réfutée**, pas seulement admirée.

> Prototype pédagogique, pas un résultat de recherche : un seul réseau, un seul domaine
> physique, trois graines aléatoires. Les limites sont listées à la fin.

## Résultats obtenus

| Mesure | Résultat |
|---|---|
| Vérification de la vérité terrain | écart équations / simulateur : `2.8e-10` p.u. |
| Curiosité vs collecte aléatoire (hors distribution, budget égal) | erreur moindre sur \|V\| sur **les trois graines** : 16,5 / 15,6 / 7,8 %. Sur les angles le gain **ne se réplique pas** (+22 / +8 / −1 %). |
| Interventions hors enveloppe de sécurité (74 expériences) | curiosité **33** · aléatoire 8 · curiosité + sécurité **6** (graine 0). L'asymétrie **se réplique** : 23–33 contre 8–11 sur trois graines. |
| Coût de la prudence | RMSE 0,00297 vs 0,00279 — quasi nul *dans ce régime* |
| Structure retrouvée sans que l'adjacence vraie soit fournie à la régression | **49/49** voisins électriques vrais sur les 6 nœuds les plus connectés ; 7 à 93 faux positifs par nœud (médiane 21) |
| Pouvoir prédictif de la loi retrouvée | R² > 0,9999 sur **127 interventions tenues à l'écart** du calage |
| Fidélité des coefficients d'admittance | R² = **0,91** contre Ybus vrai sur les 98 coefficients (conductance et susceptance de 49 arêtes) ; erreur relative médiane 2 × 10⁻⁶ |

Résultat le plus parlant : **l'agent curieux va spontanément vers les zones dangereuses**,
parce que c'est là qu'il ignore. Parmi ses 30 interventions les mieux notées, environ la
moitié (15 à 18 selon l'exécution) sortent de l'enveloppe de tension, contre 4 sur 30
tirées au hasard.

> Chaque chiffre de ce tableau est recalculé depuis les données brutes par
> `python audit_claims.py` — 13/13 vérifiés.

## Contenu

```
laplace/env.py           environnement interventionnel (pandapower, case118)
laplace/world_model.py   GNN à passage de messages + ensemble profond (incertitude)
laplace/curiosity.py     acquisition, filtre de sécurité, boucle d'apprentissage actif
laplace/distill.py       distillation symbolique (STLSQ) + score contre la vérité terrain
build_pool.py            pré-calcule le vivier d'expériences (~10 min)
fix_ood.py               (re)génère le jeu de test hors distribution
run_experiment.py        expérience complète → figures/ + results.json (~19 min)
audit_claims.py          recalcule chaque chiffre annoncé depuis les données brutes (13/13 vérifiés)
check_seeds.py           robustesse : autres graines, et test de l'excitation
make_figures.py          redessine les figures depuis results.json, sans relancer
LAPLACE_prototype.ipynb  carnet commenté (l'exécuter pour remplir les sorties)
```

## Installation et exécution

```bash
pip install pandapower torch pysindy scikit-learn matplotlib numpy pandas
python build_pool.py        # vivier d'expériences (~10 min)
python fix_ood.py           # jeu de test hors distribution (~2 min)
python run_experiment.py    # boucle complète + figures (~19 min)
jupyter notebook LAPLACE_prototype.ipynb
```

## Les quatre briques

**1. Environnement interventionnel.** Une intervention = facteurs de charge par zone
+ mise hors service d'une ligne. On peut *agir*, pas seulement observer.

**2. World model et incertitude.** Un GNN apprend (injections, topologie) → (|V|, angle).
Un ensemble profond fournit l'incertitude épistémique : là où les membres divergent,
le modèle ignore.

**3. Curiosité sous contrainte.** L'agent note chaque intervention candidate par le
désaccord de l'ensemble, puis rejette celles que le modèle juge hors enveloppe de
tension — contrainte d'ingénierie dure, pas un terme de pénalité.

**4. Des poids à la loi.** Régression parcimonieuse (STLSQ) sur un dictionnaire couvrant
**tous** les nœuds du réseau. À noter : le world model, lui, *reçoit* l'adjacence de la
topologie courante — c'est un modèle à structure de graphe. La régression, elle, ne la
reçoit pas : elle doit sélectionner elle-même les voisins électriques, et leurs coefficients.

*Trois R² distincts, à ne pas confondre* : l'ajustement en échantillon (non cité comme
résultat), le pouvoir prédictif hors échantillon (> 0,9999), et la fidélité des
coefficients contre Ybus (0,91).

## Ce que le prototype ne fait pas encore

- **Faux positifs** : la régression trouve tous les vrais voisins mais garde des termes
  parasites (7 à 93 selon le nœud, médiane 21). Le seuil de parcimonie n'est pas le bon outil.
- **Excitation insuffisante sur quelques arêtes** : l'arête médiane est retrouvée à six
  chiffres significatifs, mais une poignée d'arêtes mal excitées font tomber le R² des
  coefficients à 0,91. Coupler la conception d'expériences à la découverte de la loi.
- **Non-identifiabilité assumée** : le terme diagonal `B[i,i]` ne peut pas être retrouvé
  à partir de la seule équation en puissance active (son terme candidat vaut `V²·sin 0 = 0`).
  Il faudrait l'équation en puissance réactive. 
- **La curiosité sert le world model, pas la loi.** Coupler la conception d'expériences
  à la découverte symbolique est justement le cœur du programme proposé.
- **Le world model plafonne** : un GNN à 5 couches ne peut pas représenter une opération
  globale comme un écoulement de puissance.
- Un seul réseau, un seul domaine physique, **trois graines**. L'asymétrie de sécurité
  se réplique sur les trois ; le gain de précision est positif sur les trois mais varie
  du simple au double, et sur les angles il ne se réplique pas. `check_seeds.py` reproduit
  cette vérification.
- **Reproductibilité** : les résultats sont déterministes à environnement identique, mais
  les petits écarts bougent avec la configuration BLAS. `check_seeds.py` fixe
  `torch.set_num_threads(1)`.

---

## Licence

MIT — voir [LICENSE](LICENSE). Les données du réseau `case118` sont fournies avec
`pandapower` et relèvent des conditions propres à ce projet.

*(An English version of this document is available in [README.md](README.md).)*
