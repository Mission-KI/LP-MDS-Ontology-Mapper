import re
from typing import List, Sequence, Set, Union, cast

from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from pandas import Series

from edps.taskcontext import TaskContext

MINIMUM_NUMBER_CHARACTERS_PER_SENTENCE = 10


def _detect_language_confidences_per_sentence(detector: LanguageDetector, text: str) -> Union[Series, Language]:
    results = detector.compute_language_confidence_values(text)
    confidence = Series({result.language: result.value for result in results}, name="confidence")

    certain_language: Series[float] = confidence[confidence > 0.999]
    if len(certain_language) > 0:
        return cast(Language, certain_language.index[0])

    weighted_confidence = confidence * len(text)
    return weighted_confidence


def _calculate_language_confidences(sentences: Sequence[str]) -> tuple[Series, set[Language]]:
    language_detector = LanguageDetectorBuilder.from_all_languages().build()
    confidences = Series(
        {result.language: result.value for result in language_detector.compute_language_confidence_values("")},
        name="confidence",
    )
    certain_languages: Set[Language] = set()

    for sentence in sentences:
        certain_language_or_scores = _detect_language_confidences_per_sentence(language_detector, sentence)
        if isinstance(certain_language_or_scores, Series):
            confidences += certain_language_or_scores
        else:
            certain_languages.add(certain_language_or_scores)

    return confidences, certain_languages


def detect_languages(ctx: TaskContext, text: str) -> Set[str]:
    sentence_splitter = re.compile(r"[.:?!]\s")
    text = text.replace("\n", " ")
    sentences: List[str] = sentence_splitter.split(text)
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if (len(sentence) > MINIMUM_NUMBER_CHARACTERS_PER_SENTENCE)]
    if len(sentences) == 0:
        # Prevent accidentally filtering out all text.
        sentences = [text]

    confidences, certain_languages = _calculate_language_confidences(sentences)
    maximum_score_per_language = sum(len(sentence) for sentence in sentences)
    threshold = maximum_score_per_language * 0.15
    confidences.sort_values(ascending=False, inplace=True)
    languages = set(confidences[confidences > threshold].index)
    languages.update(certain_languages)
    ctx.logger.info("Detected Languages: %s", [language.name for language in languages])
    return {language.iso_code_639_3.name.lower() for language in languages}
