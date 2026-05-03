# MINI-PROJET

**Module :** NLP — **Formation :** M1 AAI  

**Étudiants :** HAMMADOU Islem ; MOKEDDEM Akram  

**Enseignant :** Abdoun Nabil  

---

## Objectif

Ce dépôt contient un programme Python (`nlp_pipeline.py`) qui applique les étapes classiques du traitement du langage naturel sur un fichier texte **en anglais** : pré-traitement, statistiques de n-grammes, étiquetage morpho-syntaxique (POS), reconnaissance d’entités nommées, extraction de dates par expressions régulières, et consultation lexicale via WordNet (radical, lemme, définition, exemples, synonymes, antonymes).

## Prérequis

- Python 3.8 ou supérieur  
- Connexion Internet (premier lancement : téléchargement des ressources NLTK)

## Installation

```bash
pip install -r req.txt
```

## Exécution

```bash
python main.py
```

Le script est **interactif** : il demande le chemin d’un fichier `.txt`, une liste optionnelle de stop-words supplémentaires, la valeur de **N** pour les n-grammes, puis un mot à analyser avec WordNet.

Au premier lancement, NLTK peut prendre quelques minutes pour récupérer les modèles (tokenisation, POS, chunking NER, WordNet, etc.).

## Fonctionnalités (résumé)

| Étape | Description |
|--------|-------------|
| Entrée | Fichier texte anglais choisi par l’utilisateur |
| Pré-traitement | Segmentation en phrases, tokenisation, fréquences des mots (décroissant), filtrage des stop-words (liste NLTK + liste utilisateur) |
| N-grammes | Top *N* uni-, bi- et trigrammes (sans stop-words) |
| POS tagging | Étiquettes Penn Treebank via NLTK |
| Entités nommées | `ne_chunk` (types NLTK : PERSON, ORGANIZATION, GPE, etc.) |
| Dates | Plusieurs motifs regex (ISO, formats numériques, mois en toutes lettres, trimestres, etc.) |
| Analyse lexicale | Porter (radical), lemmatisation WordNet, définition, exemples, synonymes, antonymes |

## Fichiers du projet

- `main.py` — script principal  
- `req.txt` — dépendance `nltk`  

## Note

Les résultats du NER avec NLTK sont utiles à des fins pédagogiques ; pour une précision maximale en production, on peut compléter avec un outil comme spaCy.
