# MINI-PROJET — Pipeline NLP

**Module:** NLP — **Formation:** M1 AAI

**Étudiant:** Boukhelkhal Chams Eldin

**Enseignant:** Abdoun Nabil

---

## Présentation

Ce dépôt contient un petit pipeline Python destiné au traitement de textes **en anglais**. Le programme réalise les étapes classiques suivantes : pré-traitement, calcul de n-grammes, étiquetage morpho-syntaxique (POS), reconnaissance d'entités nommées (NER), extraction de dates par expressions régulières, et consultation lexicale via WordNet (radical, lemme, définition, exemples, synonymes, antonymes).

## Prérequis

- Python 3.8 ou supérieur
- Connexion Internet (nécessaire au premier lancement pour télécharger les ressources NLTK)

## Installation

Installez les dépendances :

```bash
pip install -r req.txt
```

## Exécution (quick start)

Lancez le script principal :

```bash
python main.py
```

Le script est interactif : il demande le chemin d'un fichier `.txt`, une liste optionnelle de stop-words supplémentaires, la valeur de **N** pour l'affichage des n-grammes, puis un mot à analyser via WordNet.

Note : au premier lancement, NLTK téléchargera les modèles nécessaires (tokenizers, POS tagger, NER, WordNet, etc.). Cela peut prendre quelques minutes.

## Fonctionnalités

- Entrée : fichier texte en anglais fourni par l'utilisateur
- Pré-traitement : segmentation en phrases, tokenisation, calcul des fréquences de mots (ordre décroissant), filtrage des stop-words (liste NLTK + stop-words utilisateur)
- N-grammes : affichage des top N uni-, bi- et trigrammes (exclusion des stop-words)
- POS tagging : étiquettes Penn Treebank via NLTK
- Named Entity Recognition (NER) : sortie basée sur `ne_chunk` (PERSON, ORGANIZATION, GPE, ...)
- Extraction de dates : plusieurs motifs regex (ISO, formats numériques, mois en toutes lettres, trimestres, ...)
- Analyse lexicale : radical (Porter), lemmatisation WordNet, définition, exemples, synonymes et antonymes

## Contenu du dépôt

- [main.py](main.py) — script principal
- [req.txt](req.txt) — dépendances (ex. `nltk`)

## Remarques

Les sorties NER fournies par NLTK sont intéressantes pour l'apprentissage et les expérimentations. Pour un usage en production nécessitant une meilleure précision, envisagez d'intégrer un moteur plus robuste comme spaCy.

