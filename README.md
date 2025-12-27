# Empyree

Empyree est un projet d’exploration, d’apprentissage et de visualisation d'un espace sémantique de cartes et de decks pour le jeu Altered TCG.

## Objectif du projet

L’objectif est de comprendre la structure du jeu à travers :

- un espace sémantique des cartes (card2vec)

- un espace sémantique des decks construit à partir de celui des cartes

- des projections de ces espaces en 2 et 3 dimensions

- une exploration interactive via un dashboard

## Vue d’ensemble

Le projet est structuré comme un pipeline reproductible, allant des données brutes jusqu’à une visualisation interactive.

- Prétraitement des decks

- Entraînement des embeddings de cartes

- Construction des embeddings de decks

- Projection dans un espace commun (UMAP)

- Clustering des decks

- Exploration via un dashboard interactif

Chaque étape est isolée, testable et reproductible.

## Structure du projet

Le projet est organisé pour séparer clairement :

- le code métier

- les scripts exécutables

- les données

- les sorties liées aux modèles

- la visualisation

## Arborescence

```
.
│
├── configs/                # Configuration des modèles
│
├── data/
│   ├── raw/                # Données brutes (non versionnées)
│   └── processed/          # Données intermédiaires (non versionnées)
│
├── runs/
│   └── <model_name>/       # Outputs liés à un modèle donné (non versionnés)
│       ├── model/
│       ├── embeddings/
│       ├── projections/
│       └── clustering/
│
├── scripts/                # Scripts reproductibles
│
├── src/
│   ├── embeddings/         # Entrainement du Card2Vec
│   ├── preprocessing/      # Nettoyage et préparation des decks
│   ├── clustering/         # Embeddings, clustering, projections (UMAP)
│   └── dashboard/          # Application Dash
│
├── tests/                  # Tests unitaires
│
├── Makefile                # Liste de commandes utiles
│
├── README.md
│
├── poetry.lock
└── pyproject.toml          # Gestion des dépendances (Poetry)
```

## Gestion des modèles et des runs

Le projet distingue clairement les **données** du **résultat d’un entraînement**.

Chaque modèle entraîné correspond à un *run*, stocké dans le dossier `runs/`.
Un run regroupe l’ensemble des artefacts produits par un modèle donné :

- le modèle entraîné
- les embeddings générés
- les projections (UMAP 2D et 3D)
- les résultats de clustering

Le modèle actif est défini dans le fichier :

```bash
configs/current_model.yaml
```

Ce fichier permet de changer de modèle **sans modifier le code**, simplement en pointant vers un autre dossier de run.

Le dossier `runs/` n’est pas versionné :  
il est considéré comme une sortie expérimentale reproductible à partir du code et des données brutes.

## Projections et espace commun

Les embeddings appris (cartes et decks) vivent dans un espace vectoriel de grande dimension.
Pour permettre leur exploration visuelle, le projet utilise **UMAP** pour projeter ces espaces en 2D et en 3D.

Le choix d’UMAP permet :

- une bonne conservation des structures locales
- la possibilité de **projeter de nouveaux points** dans un espace déjà appris
- un espace commun cohérent entre cartes, decks et entrées utilisateurs

Le pipeline est le suivant :

1. L’UMAP est entraîné uniquement sur les **embeddings de cartes**
2. Les embeddings de decks sont ensuite projetés dans cet espace via `transform`
3. Les projections sont sauvegardées en 2D et 3D pour être réutilisées directement par le dashboard

Ce choix garantit que :
- les decks sont positionnés relativement aux cartes
- de nouveaux decks peuvent être ajoutés sans recalcul global
- l’espace reste stable entre les différentes visualisations

## Clustering des decks

Une fois les decks projetés dans l’espace commun, le projet cherche à identifier
des **familles de decks** partageant des similarités stratégiques.

Le clustering est réalisé avec les principes suivants :

- le clustering est effectué **séparément pour chaque faction**
- le nombre de clusters est **déterminé automatiquement** pour chaque faction
- un garde-fou permet de retomber sur un seul cluster si la structure est trop faible

La méthode utilisée repose sur :

- **K-Means** pour le clustering
- le **score de silhouette** pour estimer le nombre optimal de clusters par faction

Chaque deck se voit attribuer :
- sa faction
- un identifiant de cluster intra-faction

Ces clusters servent principalement à :
- structurer la visualisation des decks
- comparer des archétypes proches
- analyser les cartes signatures de chaque groupe

Le clustering est volontairement conservateur :  
l’objectif n’est pas de forcer des archétypes, mais de révéler des structures existantes.

## Dashboard interactif

Le projet se termine par un **dashboard interactif** permettant d’explorer visuellement
les espaces sémantiques construits tout au long du pipeline.

Le dashboard est développé avec **Dash + Plotly** et propose plusieurs espaces
d’exploration.

### Espace des cartes

Cet espace permet d’explorer l’espace sémantique des **cartes individuelles**.

Fonctionnalités principales :
- projection UMAP en **2D ou 3D**
- filtrage par **faction**
- couleur associée à la faction
- forme du point associée au type de carte
- survol affichant uniquement le nom de la carte

Cet espace sert principalement à :
- vérifier la cohérence de l’embedding card2vec
- observer les proximités entre cartes

### Espace des decks

Cet espace permet d’explorer l’espace sémantique des **decks**.

Fonctionnalités principales :
- projection UMAP en **2D ou 3D**
- filtrage par faction
- coloration par **cluster intra-faction**
- nuances de couleur pour distinguer les clusters d’une même faction

Les decks sont projetés dans le **même espace que les cartes** grâce à UMAP,
ce qui permet :
- de comparer des decks entre eux
- de situer un deck par rapport aux cartes qui le composent
- à terme, d’insérer dynamiquement un deck utilisateur dans l’espace

Le dashboard constitue le point d’entrée principal pour l’exploration du projet
et sera enrichi progressivement (détails de clusters, cartes signatures, etc.).

## Exécution du pipeline

Le projet est piloté via un Makefile qui permet d’exécuter chaque étape du pipeline de manière reproductible.

### Installation des dépendances

```bash
make install_dependencies
```

Installe l’environnement Python via Poetry.

### Pipeline complet

```bash
make all
```

Exécute l’ensemble du pipeline dans l’ordre suivant :

- Prétraitement des decks

Nettoyage et expansion des decks bruts pour construire le corpus d’entraînement.

- Entraînement des embeddings de cartes (card2vec)

Apprentissage d’un espace sémantique des cartes.

- Clustering des decks

Regroupement des decks dans l’espace des embeddings (avant réduction de dimension).

- Projection UMAP (2D et 3D)

Projection des cartes et des decks dans un espace commun pour la visualisation.

### Exécution étape par étape

Chaque étape peut être lancée indépendamment :

```bash
make preprocess   # Construction du corpus
make train        # Entraînement card2vec
make cluster      # Clustering des decks
make project      # Projections UMAP
```

### Lancer le dashboard

```bash
make run_dashboard
```

Démarre l’application Dash pour explorer :

- l’espace des cartes

- l’espace des decks

- les clusters et leurs relations

### Nettoyage des sorties

```bash
make clean
```
Supprime les sorties générées (`runs/`, caches Python) sans affecter le code ou les données sources.

## État actuel et prochaines étapes

### État actuel

À ce stade, le projet permet :

- de construire un **embedding des cartes** à partir d'un dataset de decklists (card2vec)
- de construire un **embedding des decks** par agrégation des cartes
- de projeter cartes et decks dans un **espace commun** en 2D et 3D grâce à UMAP
- de déterminer automatiquement des **clusters de decks intra-faction**
- d’explorer ces espaces via un **dashboard interactif**
- de garantir la robustesse du pipeline grâce à des **tests unitaires**

Le pipeline est reproductible, paramétrable via des fichiers de configuration,
et permet déjà une exploration qualitative riche de la structure du jeu.

### Prochaines étapes

Plusieurs axes d’amélioration sont envisagés :

- enrichir l’analyse des clusters (cartes signatures, spécificité, fréquences)
- permettre l’**insertion dynamique d’un deck utilisateur** dans l’espace
- comparer plusieurs modèles ou configurations d’embeddings
- améliorer l’ergonomie et les performances du dashboard
- documenter plus finement l’interprétation des espaces sémantiques
- envisager des méthodes de projection ou de clustering alternatives