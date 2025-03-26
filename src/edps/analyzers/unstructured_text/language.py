import re
import warnings
from collections import Counter
from typing import Iterator, List, Optional, Set, Union, cast

import spacy
import spacy.cli
import spacy.cli.download
from extended_dataset_profile.models.v0.edp import WordFrequency
from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from pandas import DataFrame, RangeIndex, Series
from spacy.tokens import Token
from spacy.tokens.doc import Doc

from edps.taskcontext import TaskContext

_TEXT_BODY_KEY = "text_body"
_BEST_LANGUAGE_KEY = "best_language"

_SANITIZER_REGEXS = {re.compile(r"\n"): " ", re.compile(r"(\s\s+)"): " "}


class MissingSpacyModelWarning(UserWarning):
    """
    Custom warning indicating that no spaCy model was found for the requested language.
    As a result, word cloud detection cannot be performed for that language.
    """


def _detect_language_confidences_per_sentence(detector: LanguageDetector, text: str) -> Series:
    results = detector.compute_language_confidence_values(text)
    confidence = Series({result.language: result.value for result in results}, name="confidence", dtype=float)
    return confidence


def _sanitize_sentence(text: str) -> str:
    for regex, replacement in _SANITIZER_REGEXS.items():
        text = regex.sub(replacement, text)
    return text


def _split_sentences(text: str) -> Iterator[str]:
    doc: Doc = spacy.blank("en")(text)
    sentencizer = spacy.pipeline.Sentencizer()
    doc = sentencizer(doc)
    for sentence in doc.sents:
        yield _sanitize_sentence(str(sentence))


def calculate_language_confidences(text: str) -> tuple[DataFrame, set[Language]]:
    sentences = list(_split_sentences(text))
    if len(sentences) == 0:
        # Prevent accidentally filtering out all text.
        sentences = [_sanitize_sentence(text)]

    language_detector = LanguageDetectorBuilder.from_all_languages().build()
    columns: List[Union[Language, str]] = [
        result.language for result in language_detector.compute_language_confidence_values("")
    ]
    columns += [_TEXT_BODY_KEY]
    confidences = DataFrame(columns=columns, index=RangeIndex(0, 0))
    certain_languages: Set[Language] = set()

    for index, sentence in enumerate(sentences):
        current_sentence_scores = _detect_language_confidences_per_sentence(language_detector, sentence)
        confidences.loc[index] = current_sentence_scores
        confidences[_TEXT_BODY_KEY] = confidences[_TEXT_BODY_KEY].astype(object)
        confidences.loc[index, _TEXT_BODY_KEY] = sentence

        certain_language: Series[float] = current_sentence_scores[current_sentence_scores > 0.999]
        if len(certain_language) > 0:
            certain_languages.add(certain_language.index[0])

    return confidences, certain_languages


def detect_languages(ctx: TaskContext, text: str) -> Set[str]:
    confidences, certain_languages = calculate_language_confidences(text)
    # Weight confidence by sentence length!
    sentence_lengths = confidences[_TEXT_BODY_KEY].str.len()
    weighted_confidences = confidences.drop(_TEXT_BODY_KEY, axis=1).mul(sentence_lengths, axis=0)
    weighted_confidence_sums = weighted_confidences.sum()
    # Apply threshold
    maximum_score_per_language = sentence_lengths.sum()
    threshold = maximum_score_per_language * 0.15
    weighted_confidence_sums.sort_values(ascending=False, inplace=True)
    languages = set(weighted_confidence_sums[weighted_confidence_sums > threshold].index)
    languages.update(certain_languages)
    ctx.logger.info("Detected Languages: %s", [language.name for language in languages])
    return {language.iso_code_639_3.name.lower() for language in languages}


def detect_word_cloud(ctx: TaskContext, text: str, top_n: int = 10) -> Iterator[WordFrequency]:
    aggregated_counts: Counter[str] = Counter()
    confidences, _ = calculate_language_confidences(text)
    confidences[_BEST_LANGUAGE_KEY] = confidences.iloc[:, :-1].idxmax(axis=1)
    text_by_lang = confidences.groupby(_BEST_LANGUAGE_KEY)[_TEXT_BODY_KEY].apply(". ".join)

    for lang, text in text_by_lang.items():
        nlp = _load_spacy_model(ctx, cast(Language, lang))

        if nlp is None:
            continue

        doc = nlp(text)
        word_counts = Counter(
            _normalize_token(token) for token in doc if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 3
        )
        aggregated_counts.update(word_counts)

    for word, count in aggregated_counts.most_common(top_n):
        yield WordFrequency(word=word, count=count)


LANGUAGE_MODEL_MAP = {
    Language.GERMAN: "de_core_news_lg",
    Language.ENGLISH: "en_core_web_lg",
    Language.FRENCH: "fr_core_news_lg",
}


def _load_spacy_model(ctx: TaskContext, lang: Language) -> Optional[spacy.language.Language]:
    model_name = LANGUAGE_MODEL_MAP.get(lang)
    if not model_name:
        message = f"Can not detect word cloud for text. No spaCy model found for language '{lang}'."
        ctx.logger.warning(message)
        warnings.warn(message, MissingSpacyModelWarning)
        return None

    nlp = spacy.load(model_name)
    nlp.max_length = 5000000
    return nlp


def _normalize_token(token: Token, lemma_tags: Set[str] = {"NNS", "NNPS"}):
    """
    Normalize a spaCy token by converting plural forms to singular.

    Default tags:
    - "NNS": plural common noun (e.g., "cars")
    - "NNPS": plural proper noun (e.g., "Smiths")
    """
    return token.lemma_ if token.tag_ in lemma_tags else token.text


def download_supported_languages():
    for model in LANGUAGE_MODEL_MAP.values():
        if not spacy.util.is_package(model):
            spacy.cli.download(model)
