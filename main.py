#HAMMADOU islem , MOKEDDEM AKRAM
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import nltk
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree


def ensure_nltk_data() -> None:
    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "wordnet",
        "omw-1.4",
    ]
    for pkg in packages:
        nltk.download(pkg, quiet=True)


def read_text_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {p}")
    return p.read_text(encoding="utf-8", errors="replace")


def segment_sentences(text: str) -> List[str]:
    return sent_tokenize(text)


def tokenize_words(text: str) -> List[str]:
    return word_tokenize(text)


def word_frequencies(tokens: Sequence[str]) -> List[Tuple[str, int]]:
    words = [t.lower() for t in tokens if t.isalpha()]
    return Counter(words).most_common()


def load_stopwords_standard(lang: str = "english") -> set:
    return set(stopwords.words(lang))


def remove_stopwords(
    tokens: Sequence[str],
    standard_stops: Iterable[str],
    user_stops: Iterable[str],
) -> List[str]:
    std = {w.lower() for w in standard_stops}
    usr = {w.lower() for w in user_stops}
    combined = std | usr
    return [t.lower() for t in tokens if t.isalpha() and t.lower() not in combined]


def top_ngrams(tokens: Sequence[str], n: int, top_n: int) -> List[Tuple[Tuple[str, ...], int]]:
    if len(tokens) < n:
        return []
    if n == 1:
        counts = Counter(tokens)
    else:
        counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return counts.most_common(top_n)


def penn_to_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return wn.NOUN


def pos_tag_tokens(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return pos_tag(list(tokens))


def extract_named_entities(tagged: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    tree = ne_chunk(tagged)
    out: List[Tuple[str, str]] = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            label = subtree.label()
            phrase = " ".join(token for token, _ in subtree.leaves())
            out.append((phrase, label))
    return out


DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?\b",
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\s*,\s*\d{4}\b",
    r"\b(?:Q[1-4]\s*['\u2019]?\s*\d{4}|FY\s*\d{4})\b",
    r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
]


def compile_date_regex() -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in DATE_PATTERNS)
    return re.compile(combined, re.IGNORECASE)


def extract_dates(text: str, pattern: re.Pattern | None = None) -> List[str]:
    pat = pattern or compile_date_regex()
    matches = pat.findall(text)
    flat = [m if isinstance(m, str) else next(filter(None, m)) for m in matches]
    return sorted(set(flat))


def wordnet_lexical_info(word: str, pos_tag_token: str | None = None) -> dict:
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    wn_pos = penn_to_wordnet_pos(pos_tag_token or "NN")

    stem = stemmer.stem(word.lower())
    lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)

    synsets = wn.synsets(word.lower(), pos=wn_pos) or wn.synsets(word.lower())
    if not synsets:
        return {
            "radical": stem,
            "lemme": lemma,
            "definition": None,
            "exemples": [],
            "synonymes": [],
            "antonymes": [],
            "note": "Aucun synset WordNet pour ce mot",
        }

    s0 = synsets[0]
    definition = s0.definition()
    examples = list(s0.examples())

    synonyms = set()
    antonyms = set()
    for syn in synsets[:5]:
        for lem in syn.lemmas():
            synonyms.add(lem.name().replace("_", " "))
            for ant in lem.antonyms():
                antonyms.add(ant.name().replace("_", " "))

    synonyms.discard(word.lower())
    return {
        "radical": stem,
        "lemme": lemma,
        "definition": definition,
        "exemples": examples,
        "synonymes": sorted(synonyms),
        "antonymes": sorted(antonyms),
    }


def prompt_user_stopwords() -> List[str]:
    print("Mots vides utilisateur (séparés par des virgules, Entrée pour ignorer) : ", end="")
    line = input().strip()
    if not line:
        return []
    return [w.strip() for w in line.split(",") if w.strip()]


def main() -> None:
    ensure_nltk_data()

    print("=== Pipeline NLP ===\n")
    path = "test.txt"
    text = read_text_file(path)

    sentences = segment_sentences(text)
    print(f"\nPhrases: {len(sentences)}")

    raw_tokens = tokenize_words(text)
    print(f"Tokens sample: {raw_tokens[:30]}")

    freq_all = word_frequencies(raw_tokens)
    print("\nTop words:")
    for w, c in freq_all[:15]:
        print(w, c)

    std_stops = load_stopwords_standard()
    user_stops = prompt_user_stopwords()
    filtered = remove_stopwords(raw_tokens, std_stops, user_stops)

    print(f"\nTokens after stopwords: {len(filtered)}")

    n_str = input("N (default 10): ").strip()
    top_n = int(n_str) if n_str.isdigit() else 10

    uni = top_ngrams(filtered, 1, top_n)
    bi = top_ngrams(filtered, 2, top_n)
    tri = top_ngrams(filtered, 3, top_n)

    print("\nUnigrams:")
    for g, c in uni:
        print(g[0], c)

    print("\nBigrams:")
    for g, c in bi:
        print(" ".join(g), c)

    print("\nTrigrams:")
    for g, c in tri:
        print(" ".join(g), c)

    tagged_full = pos_tag_tokens(raw_tokens)
    print("\nPOS sample:")
    print(tagged_full[:20])

    entities = extract_named_entities(tagged_full)
    print("\nEntities:")
    for phrase, label in entities[:20]:
        print(label, phrase)

    dates = extract_dates(text)
    print("\nDates:")
    for d in dates:
        print(d)

    word = input("\nWord: ").strip()
    if word:
        w0 = word_tokenize(word)[0]
        tag_one = pos_tag_tokens([w0])[0][1]
        info = wordnet_lexical_info(w0, tag_one)
        print("\nStem:", info["radical"])
        print("Lemma:", info["lemme"])
        print("Definition:", info["definition"])
        print("Examples:", info["exemples"])
        print("Synonyms:", info["synonymes"])
        print("Antonyms:", info["antonymes"])

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
