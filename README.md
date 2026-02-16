# 🛍️ Système de Recommandation E-Commerce Multimodal

Système de recommandation hybride pour le e-commerce fashion, combinant **vision par ordinateur** (CLIP), **filtrage collaboratif** (ALS), **embeddings textuels** (Sentence-Transformers) et **explicabilité** (SHAP).

> **Dataset** : Amazon Fashion — 6 441 articles, 5 000 utilisateurs, 18 266 interactions.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Streamlit                       │
│         app.py — 3 onglets (Hybride · Image · ALS)             │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / JSON
┌────────────────────────▼────────────────────────────────────────┐
│                     API FastAPI (port 8001)                     │
│                    src/api/amazon_api.py                        │
│   /health · /amazon/recommend-hybrid · /amazon/similar-items   │
│   /amazon/recommend-user · /amazon/explain-recommendation      │
└──┬──────────────┬──────────────┬──────────────┬────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
 CLIP          ALS (implicit)  Sentence-     SHAP +
 ViT-B/32     user/item        Transformers  Surrogate RF
 (images)     factors          (MiniLM-L6)   (explainability)
   │              │              │
   ▼              ▼              ▼
 FAISS         CSR matrix     Embeddings
 IndexFlatIP   (sparse)       textuels
```

### Moteurs de recommandation

| Moteur | Description | Signal |
|--------|-------------|--------|
| **Hybrid** | Fusion pondérée de 3 signaux : `α·image + β·ALS + γ·popularité` | Image + Collaboratif + Popularité |
| **Multimodal KNN** | Recherche par similarité dans l'espace d'embeddings fusionnés (CLIP + texte) via FAISS | Image + Texte |
| **ALS** | Filtrage collaboratif (Alternating Least Squares) sur la matrice user-item | Interactions utilisateur |
| **Popularité** | Baseline — recommande les articles les plus populaires | Comptage d'interactions |

### Explicabilité (3 méthodes)

Le système offre **3 niveaux d'explicabilité** complémentaires :

| Méthode | Type | Description |
|---------|------|-------------|
| **SHAP (local)** | Per-recommendation | Décompose chaque recommandation en contributions (image, ALS, popularité) via un modèle Random Forest surrogate |
| **Counterfactual** | Contrastif | "Si on retirait le signal X, ce produit passerait du rang 2 au rang 8" — analyse de sensibilité par signal |
| **Global Explanations** | Vue d'ensemble | Importance globale des features, distribution de confiance du modèle, patterns du dataset |

---

## 📁 Structure du projet

```
ecommerce-reco/
├── frontend/
│   └── app.py                  # Interface Streamlit (4 onglets)
├── src/
│   ├── api/
│   │   └── amazon_api.py       # API FastAPI — tous les endpoints
│   ├── config/
│   │   ├── domain_adapter.py   # Classe abstraite DomainAdapter
│   │   ├── yaml_adapter.py     # Implémentation YAML du DomainAdapter
│   │   ├── settings.py         # Chargeur de configuration (singleton)
│   │   └── domains/            # Fichiers YAML par domaine
│   │       ├── ecommerce.yaml  # E-Commerce Fashion (défaut)
│   │       ├── healthcare.yaml # Santé
│   │       └── education.yaml  # Éducation
│   ├── recommenders/
│   │   ├── hybrid.py           # Moteur hybride (fusion des scores)
│   │   └── multimodal.py       # Fusion d'embeddings image + texte
│   ├── encoders/               # Encodeurs CLIP & texte
│   ├── explain/
│   │   ├── shap_surrogate.py   # Entraînement du surrogate RF + SHAP
│   │   ├── counterfactual.py   # Raisonnement contrefactuel
│   │   └── global_explain.py   # Explications globales (importance, confiance, patterns)
│   ├── models/                 # Modèle ALS (implicit)
│   └── utils/                  # Utilitaires (images, paths, etc.)
├── scripts/
│   ├── evaluate_all_metrics.py # Évaluation complète (métriques IR + catégorielles + SHAP)
│   ├── test_full_system.py     # Tests automatisés (207 tests)
│   ├── build_*.py              # Scripts de construction (embeddings, FAISS, etc.)
│   └── train_*.py              # Scripts d'entraînement (ALS, surrogate)
├── data/
│   ├── amazon/processed_small/ # Données traitées (items, interactions, ALS)
│   ├── embeddings/             # Embeddings multimodaux + index FAISS
│   └── models/                 # Modèle surrogate RF (surrogate_rf.joblib)
├── reports/                    # Rapports d'évaluation (CSV)
└── requirements.txt
```

---

## 🌐 Architecture Domain-Agnostic

Le système est conçu avec une **couche d'abstraction de domaine** permettant de réutiliser les mêmes moteurs de recommandation et d'explicabilité sur **n'importe quel domaine applicatif** — sans modifier le code source.

### Principe

Un **DomainAdapter** abstrait sert de contrat entre le code générique (moteurs, API, frontend) et un fichier de configuration YAML spécifique au domaine :

```
┌────────────────────────────┐
│  Code applicatif (API,     │
│  moteurs, explicabilité)   │
└────────────┬───────────────┘
             │  appelle
┌────────────▼───────────────┐
│   DomainAdapter (abstrait) │
│   - load_items()           │
│   - get_column_map()       │
│   - entity_labels()        │
│   - explain_reason()       │
│   - get_paths()            │
│   - get_engine_defaults()  │
└────────────┬───────────────┘
             │  implémenté par
┌────────────▼───────────────┐
│  YAMLDomainAdapter         │
│  lit src/config/domains/   │
│     └── <domaine>.yaml     │
└────────────────────────────┘
```

### Mapping des concepts par domaine

| Concept générique | E-Commerce (défaut) | Santé | Éducation |
|-------------------|---------------------|-------|-----------|
| **Utilisateur** | Acheteur (`user_id`) | Patient (`patient_id`) | Étudiant (`student_id`) |
| **Item** | Produit (`item_idx`) | Traitement (`treatment_id`) | Cours (`course_id`) |
| **Interaction** | Achat / note | Prescription / efficacité | Inscription / complétion |
| **Catégorie** | Sous-catégorie mode | Spécialité médicale | Matière |
| **Explication** | "Recommandé car visuellement similaire..." | "Suggéré car efficace pour des profils similaires..." | "Proposé car des étudiants similaires ont suivi..." |

### Changer de domaine

Le domaine actif est contrôlé par la variable d'environnement `RECO_DOMAIN` :

```powershell
# Utiliser le domaine e-commerce (défaut)
$env:RECO_DOMAIN = "ecommerce"
python -m uvicorn src.api.amazon_api:app --port 8001

# Utiliser le domaine santé
$env:RECO_DOMAIN = "healthcare"
python -m uvicorn src.api.amazon_api:app --port 8001
```

### Ajouter un nouveau domaine

1. Créer `src/config/domains/<nouveau_domaine>.yaml` en suivant le schéma existant (voir `ecommerce.yaml`)
2. Placer les données dans les chemins déclarés dans le YAML
3. Lancer avec `RECO_DOMAIN=<nouveau_domaine>`

### Fichiers de configuration disponibles

| Fichier | Domaine | Description |
|---------|---------|-------------|
| `src/config/domains/ecommerce.yaml` | E-Commerce Fashion | Configuration par défaut — Amazon Fashion |
| `src/config/domains/healthcare.yaml` | Santé | Recommandation de traitements médicaux |
| `src/config/domains/education.yaml` | Éducation | Recommandation de cours en ligne |

### Endpoint `/domain`

L'API expose un endpoint `GET /domain` qui retourne la configuration active :

```json
{
  "active_domain": "ecommerce",
  "display_name": "E-Commerce Fashion",
  "entities": {"user": "Acheteur", "item": "Produit", "interaction": "Achat"},
  "column_mapping": {"item_id": "item_idx", "title": "title", "category": "main_category"},
  "engine_defaults": {"default_engine": "hybrid", "hybrid_weights": {"alpha": 0.5, "beta": 0.4, "gamma": 0.1}},
  "available_domains": ["ecommerce", "education", "healthcare"]
}
```

---

## 🚀 Installation & Lancement

### Prérequis

- **Python 3.10+**
- **Git**
- Les données dans `data/` (images, embeddings, modèles pré-entraînés)

### 1. Cloner et installer

```powershell
git clone https://github.com/<votre-repo>/ecommerce-reco.git
cd ecommerce-reco

# Créer l'environnement virtuel
python -m venv .venv
& .venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Récupérer les données

Les données (images, embeddings, modèles pré-entraînés) ne sont pas incluses dans le dépôt Git en raison de leur taille. Téléchargez-les depuis Google Drive :

📥 **[Télécharger le dossier `data/`](https://drive.google.com/drive/folders/1x6aDZpV0AxBL6bcpnEXBL9WA6qOxUFZH?usp=sharing)**

Placez le contenu téléchargé dans le répertoire `data/` à la racine du projet :

```
ecommerce-reco/
└── data/
    ├── amazon/processed_small/   # Items, interactions, matrices ALS
    ├── embeddings/               # Embeddings multimodaux + index FAISS
    └── models/                   # Modèle surrogate RF (surrogate_rf.joblib)
```

### 3. Lancer l'API FastAPI

```powershell
python -m uvicorn src.api.amazon_api:app --host 127.0.0.1 --port 8001 --reload
```

L'API est disponible sur `http://127.0.0.1:8001`. Vérifier avec :
```
GET http://127.0.0.1:8001/health
```
4
### 3. Lancer l'interface Streamlit

Dans un **second terminal** :

```powershell
streamlit run frontend/app.py
```

L'interface s'ouvre sur `http://localhost:8501` avec 3 onglets :
- **Hybride** — Recommandation hybride (moteur principal)
- **Similarité Image** — Recherche par similarité visuelle (CLIP)
- **Utilisateur ALS** — Recommandations personnalisées par filtrage collaboratif
5
### 4. Lancer les tests

```powershell
python scripts/test_full_system.py
```
> ✅ 207 tests / 0 échecs
6
### 5. Lancer l'évaluation des métriques

```powershell
python scripts/evaluate_all_metrics.py --split test
```
> Génère `reports/evaluation_metrics_all.csv`

---

## 📊 Résultats d'évaluation

### Métriques d'interaction utilisateur

Évaluation sur 665 utilisateurs (split test, seuil ≥ 4.0). L'objectif est de retrouver les articles effectivement achetés/notés par chaque utilisateur.

| Métrique | ALS | Popularité (baseline) | Gain ALS vs baseline |
|----------|----:|-----:|:----:|
| **Precision@10** | 0.62% | 0.35% | **+78%** |
| **Recall@10** | 6.17% | 3.46% | **+78%** |
| **Recall@20** | 9.17% | 6.62% | **+39%** |
| **NDCG@10** | 3.04% | 1.82% | **+67%** |
| **NDCG@20** | 3.79% | 2.64% | **+44%** |
| **MRR** | 2.30% | 1.57% | **+47%** |
| **MAP@10** | 2.10% | 1.33% | **+58%** |
| **HitRate@20** | 9.17% | 6.62% | **+39%** |
| **Coverage** | **57.5%** | 0.4% | **×143** |

> **ALS surpasse le baseline Popularité** sur toutes les métriques, avec une couverture (diversité) 143× supérieure.
>
> Les valeurs absolues basses sont attendues : le dataset a une densité de 0.056% (matrice très creuse) avec ~1.25 items pertinents par utilisateur dans le test set — ce qui est typique des datasets e-commerce réels.

### Cohérence catégorielle

Évaluation sur 1 000 items requêtes (41 sous-catégories extraites : ring, dress, sunglasses, socks, etc.). Mesure si les recommandations appartiennent à la même sous-catégorie que l'item requête.

| Métrique | Multimodal KNN | Hybrid | Random (baseline) |
|----------|------:|------:|------:|
| **Cat-Precision@5** | **6.58%** | 3.24% | 5.34% |
| **Cat-Precision@10** | **6.52%** | 3.64% | 6.28% |
| **Cat-HitRate@5** | **26.8%** | 12.8% | — |
| **Cat-HitRate@10** | **42.4%** | 27.8% | — |
| **Cat-HitRate@20** | **60.6%** | 52.3% | — |

> Le **Multimodal KNN** recommande des articles de la même sous-catégorie significativement mieux que le tirage aléatoire. Dans le top 20, un article pertinent est trouvé **6 fois sur 10**.
>
> Le **Hybrid** propose un bon compromis diversité/pertinence : **52.3%** de Cat-HitRate@20 avec une couverture de **51.2%**, grâce à la fusion avec ALS et popularité.

### Modèle Surrogate — Explicabilité SHAP

| Métrique | Valeur |
|----------|--------|
| Feature Importance RF — `multimodal_cosine` | **99.97%** |
| Feature Importance RF — `als_dot` | 0.025% |
| Feature Importance RF — `popularity` | ~0% |
| Mean\|SHAP\| — `multimodal_cosine` | **0.036** |
| Mean\|SHAP\| — `als_dot` | 0.0005 |

> La similarité visuelle (cosine CLIP) est le signal dominant, confirmé par les valeurs SHAP. Le modèle surrogate fournit des **explications interprétables** pour chaque recommandation (barres de contribution dans l'interface).

---

## 🔌 Principaux endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Vérification de l'état de l'API |
| `/amazon/item/{id}` | GET | Métadonnées d'un article |
| `/amazon/image/{id}` | GET | Image d'un article |
| `/amazon/sample-items` | GET | Échantillon aléatoire d'articles |
| `/amazon/similar-items` | GET | Recommandations par similarité d'embeddings (KNN) |
| `/amazon/recommend-user` | GET | Recommandations ALS personnalisées |
| `/amazon/recommend-hybrid` | GET | Recommandations hybrides (image + ALS + popularité) |
| `/amazon/explain-recommendation` | GET | Explication SHAP d'une recommandation |
| `/amazon/counterfactual` | GET | Analyse contrefactuelle (impact de chaque signal) |
| `/amazon/global-explanations` | GET | Explications globales (importance, confiance, patterns) |
| `/amazon/feedback` | POST | Collecte de feedback utilisateur |
| `/domain` | GET | Configuration du domaine actif |

---

## ⚙️ Technologies

| Composant | Technologie |
|-----------|-------------|
| Embeddings image | **CLIP** (openai/clip-vit-base-patch32) — 512 dimensions |
| Embeddings texte | **Sentence-Transformers** (all-MiniLM-L6-v2) — 384 dimensions |
| Filtrage collaboratif | **ALS** (implicit) — 64 facteurs latents |
| Recherche de voisins | **FAISS** (IndexFlatIP) |
| Explicabilité | **SHAP** + Random Forest surrogate |
| API Backend | **FastAPI** + Uvicorn |
| Frontend | **Streamlit** |
| Données | **Amazon Fashion** (reviews & metadata) |

---

## 📝 Notes techniques

- **Normalisation ALS** : les scores ALS bruts (dot-product ~200+) sont normalisés via une sigmoïde `1/(1+exp(-x/30))` pour un affichage équilibré avec les scores cosine ∈ [0, 1].
- **Embeddings multimodaux** : fusion des vecteurs CLIP (image) et MiniLM (texte) en un vecteur 512-d unique, indexé dans FAISS.
- **Sparse dataset** : densité de 0.056% — le système est conçu pour fonctionner dans des conditions de cold-start réalistes.
- **Tests** : 207 tests automatisés couvrant tous les endpoints et cas limites (`scripts/test_full_system.py`).
