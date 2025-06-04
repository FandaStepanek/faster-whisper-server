"""Word processing utilities for transcription enhancement."""

from dataclasses import dataclass
import logging
import re
import unicodedata
from typing import Optional, Tuple

from rapidfuzz import process as rapidfuzz_process
from rapidfuzz.distance import Levenshtein

from speaches.api_types import TranscriptionWord

logger = logging.getLogger(__name__)

def normalize_word(word: str, remove_punctuation: bool = True) -> str:
    """Normalize word by removing diacritics and optionally punctuation.
    
    Args:
        word: Word to normalize
        remove_punctuation: Whether to remove punctuation marks
        
    Returns:
        Normalized word
    """
    # Remove diacritics
    nfkd_form = unicodedata.normalize('NFKD', word)
    word_without_diacritics = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    if remove_punctuation:
        # Remove punctuation and whitespace
        word_clean = re.sub(r'[^\w\s]', '', word_without_diacritics)
    else:
        word_clean = word_without_diacritics
    
    # Remove extra whitespace and convert to lowercase
    return word_clean.strip().lower()

@dataclass
class WordContext:
    """Context information for word processing.
    
    Attributes:
        prev_word: Previous word in the segment, if any
        next_word: Next word in the segment, if any
        position_in_segment: Index of current word in segment
        total_words: Total number of words in segment
        avg_segment_confidence: Average confidence score across segment
    """
    prev_word: Optional[TranscriptionWord]
    next_word: Optional[TranscriptionWord]
    position_in_segment: int
    total_words: int
    avg_segment_confidence: float


def get_word_context(
    words: list[TranscriptionWord],
    current_idx: int
) -> WordContext:
    """Get context information for a word in a segment.
    
    Args:
        words: List of all words in the segment
        current_idx: Index of the current word
        
    Returns:
        WordContext object containing contextual information
    """
    context = WordContext(
        prev_word=words[current_idx - 1] if current_idx > 0 else None,
        next_word=words[current_idx + 1] if current_idx < len(words) - 1 else None,
        position_in_segment=current_idx,
        total_words=len(words),
        avg_segment_confidence=sum(w.probability for w in words) / len(words)
    )
    logger.debug(
        "Word context: pos=%d/%d, avg_conf=%.3f, prev='%s', next='%s'",
        context.position_in_segment,
        context.total_words,
        context.avg_segment_confidence,
        context.prev_word.word if context.prev_word else None,
        context.next_word.word if context.next_word else None
    )
    return context


def should_process_word(
    word: TranscriptionWord,
    context: WordContext,
    confidence_threshold: float,
    hotwords: list[str] | None = None
) -> bool:
    """Determine if a word should be processed based on various factors.
    
    Args:
        word: The word to check
        context: Contextual information about the word
        confidence_threshold: Base confidence threshold
        hotwords: List of hotwords to check for potential matches
        
    Returns:
        True if the word should be processed, False otherwise
    """
    logger.debug(
        "Checking word '%s' (conf=%.3f) against threshold %.3f",
        word.word,
        word.probability,
        confidence_threshold
    )
    
    # If confidence is very high, only process if there's a potential exact hotword match
    if word.probability >= confidence_threshold:
        if hotwords:
            word_norm = normalize_word(word.word)
            # Check for exact matches or potential compound words
            for hotword in hotwords:
                hotword_norm = normalize_word(hotword)
                if (word_norm in hotword_norm or 
                    (context.next_word and 
                     normalize_word(f"{word.word} {context.next_word.word}") in hotword_norm)):
                    logger.debug(
                        "Word '%s' (normalized: '%s') has high confidence but potential hotword match with '%s' (normalized: '%s')",
                        word.word,
                        word_norm,
                        hotword,
                        hotword_norm
                    )
                    return True
        logger.debug("Word '%s' above confidence threshold and no potential hotword matches, skipping", word.word)
        return False
        
    # Always process words with very low confidence
    if word.probability < 0.5:
        logger.debug("Word '%s' has very low confidence, will process", word.word)
        return True
        
    # If it's significantly lower confidence than segment average
    if word.probability < context.avg_segment_confidence * 0.8:  # Relaxed from 0.7
        logger.debug(
            "Word '%s' confidence (%.3f) lower than segment average (%.3f)",
            word.word,
            word.probability,
            context.avg_segment_confidence
        )
        return True
        
    # If neighboring words have higher confidence
    if (context.prev_word and context.next_word and
        word.probability < min(context.prev_word.probability, context.next_word.probability) * 0.9):  # Relaxed from 0.8
        logger.debug(
            "Word '%s' confidence (%.3f) lower than neighbors (prev=%.3f, next=%.3f)",
            word.word,
            word.probability,
            context.prev_word.probability,
            context.next_word.probability
        )
        return True
        
    # If it's short duration compared to neighbors
    if context.prev_word and context.next_word:
        word_duration = word.end - word.start
        prev_duration = context.prev_word.end - context.prev_word.start
        next_duration = context.next_word.end - context.next_word.start
        avg_neighbor_duration = (prev_duration + next_duration) / 2
        if word_duration < avg_neighbor_duration * 0.6:  # Relaxed from 0.5
            logger.debug(
                "Word '%s' duration (%.3f) shorter than neighbors avg (%.3f)",
                word.word,
                word_duration,
                avg_neighbor_duration
            )
            return True
            
    # Check for potential fuzzy matches with hotwords
    if hotwords:
        word_norm = normalize_word(word.word)
        for hotword in hotwords:
            hotword_norm = normalize_word(hotword)
            # Use a simple substring check as a quick pre-filter
            if (len(word_norm) >= 3 and  # Only check words of reasonable length
                (word_norm in hotword_norm or hotword_norm in word_norm or
                 any(w in hotword_norm for w in word_norm.split()))):
                logger.debug(
                    "Word '%s' (normalized: '%s') has potential fuzzy match with hotword '%s' (normalized: '%s')",
                    word.word,
                    word_norm,
                    hotword,
                    hotword_norm
                )
                return True
            
    logger.debug("Word '%s' does not need processing", word.word)
    return False


def find_best_hotword_match(
    word: TranscriptionWord,
    context: WordContext,
    hotwords: list[str],
    score_cutoff: float
) -> Tuple[Optional[str], float]:
    """Find the best matching hotword considering context.
    
    Args:
        word: The word to match
        context: Contextual information about the word
        hotwords: List of hotwords to match against
        score_cutoff: Minimum similarity score (0-100)
        
    Returns:
        Tuple of (best matching hotword or None, match score)
    """
    logger.debug(
        "Finding best match for word '%s' among %d hotwords (cutoff=%.1f)",
        word.word,
        len(hotwords),
        score_cutoff
    )
    
    # Extract punctuation from the word to preserve it
    word_text = word.word.strip()
    punctuation_suffix = ''
    if word_text and word_text[-1] in '.?!,;:':
        punctuation_suffix = word_text[-1]
        word_text = word_text[:-1].strip()
    
    # Create a mapping of normalized hotwords to their original forms
    hotword_map = {normalize_word(hw): hw for hw in hotwords}
    
    # Try exact match first
    word_norm = normalize_word(word_text)
    if word_norm in hotword_map:
        result = hotword_map[word_norm] + punctuation_suffix
        logger.debug(
            "Found exact match: '%s' (normalized from '%s', original hotword: '%s')", 
            result, 
            word.word,
            hotword_map[word_norm]
        )
        return result, 1.0
            
    # Check for compound words that might have been split
    if context.next_word:
        compound_norm = normalize_word(f"{word_text} {context.next_word.word}")
        for norm_hotword, orig_hotword in hotword_map.items():
            if compound_norm == norm_hotword:
                result = orig_hotword.split()[0] + punctuation_suffix
                logger.debug(
                    "Found compound word match: '%s' (normalized from '%s', original hotword: '%s')",
                    result,
                    word.word,
                    orig_hotword
                )
                return result, 1.0
                
    # Try fuzzy matching with context-aware scoring
    matches = []
    for norm_hotword, orig_hotword in hotword_map.items():
        # Basic fuzzy match score
        match = rapidfuzz_process.extractOne(
            word_norm,
            [norm_hotword],
            scorer=Levenshtein.normalized_similarity,
            score_cutoff=score_cutoff/100
        )
        
        if match:
            base_score = match[1]
            context_score = 1.0
            
            # Boost score if neighboring words support this match
            if context.prev_word and context.next_word:
                # Check if this hotword commonly appears with neighboring words
                neighbor_match = max(
                    rapidfuzz_process.extractOne(
                        normalize_word(f"{context.prev_word.word} {orig_hotword}"),
                        [norm_hotword],
                        scorer=Levenshtein.normalized_similarity,
                        score_cutoff=0
                    )[1],
                    rapidfuzz_process.extractOne(
                        normalize_word(f"{orig_hotword} {context.next_word.word}"),
                        [norm_hotword],
                        scorer=Levenshtein.normalized_similarity,
                        score_cutoff=0
                    )[1]
                )
                context_score += neighbor_match * 0.2
            
            final_score = base_score * context_score
            if final_score >= score_cutoff/100:
                result = orig_hotword + punctuation_suffix
                logger.debug(
                    "Found fuzzy match: '%s' -> '%s' (score=%.3f, original hotword: '%s')",
                    word.word,
                    result,
                    final_score,
                    orig_hotword
                )
                matches.append((result, final_score))
    
    if matches:
        # Return the match with highest score
        best_match = max(matches, key=lambda x: x[1])
        logger.debug("Best match: '%s' (score=%.3f)", best_match[0], best_match[1])
        return best_match
        
    logger.debug("No matches found for word '%s'", word.word)
    return None, 0.0 