"""Word processing utilities for transcription enhancement."""

from dataclasses import dataclass
from typing import Optional, Tuple

from rapidfuzz import process as rapidfuzz_process
from rapidfuzz.distance import Levenshtein

from speaches.api_types import TranscriptionWord


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
    return WordContext(
        prev_word=words[current_idx - 1] if current_idx > 0 else None,
        next_word=words[current_idx + 1] if current_idx < len(words) - 1 else None,
        position_in_segment=current_idx,
        total_words=len(words),
        avg_segment_confidence=sum(w.probability for w in words) / len(words)
    )


def should_process_word(
    word: TranscriptionWord,
    context: WordContext,
    confidence_threshold: float
) -> bool:
    """Determine if a word should be processed based on various factors.
    
    Args:
        word: The word to check
        context: Contextual information about the word
        confidence_threshold: Base confidence threshold
        
    Returns:
        True if the word should be processed, False otherwise
    """
    # Basic confidence check
    if word.probability >= confidence_threshold:
        return False
        
    # If it's significantly lower confidence than segment average
    if word.probability < context.avg_segment_confidence * 0.7:
        return True
        
    # If both neighboring words have much higher confidence
    if (context.prev_word and context.next_word and
        word.probability < min(context.prev_word.probability, context.next_word.probability) * 0.8):
        return True
        
    # If it's very short duration compared to neighbors
    if context.prev_word and context.next_word:
        word_duration = word.end - word.start
        prev_duration = context.prev_word.end - context.prev_word.start
        next_duration = context.next_word.end - context.next_word.start
        avg_neighbor_duration = (prev_duration + next_duration) / 2
        if word_duration < avg_neighbor_duration * 0.5:
            return True
            
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
    # Try exact match first
    word_lower = word.word.lower()
    for hotword in hotwords:
        if word_lower == hotword.lower():
            return hotword, 1.0
            
    # Check for compound words that might have been split
    if context.next_word:
        compound = f"{word.word} {context.next_word.word}".lower()
        for hotword in hotwords:
            if compound == hotword.lower():
                return hotword.split()[0], 1.0
                
    # Try fuzzy matching with context-aware scoring
    matches = []
    for hotword in hotwords:
        # Basic fuzzy match score
        match = rapidfuzz_process.extractOne(
            word.word,
            [hotword],
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
                        f"{context.prev_word.word} {hotword}",
                        [hotword],
                        scorer=Levenshtein.normalized_similarity,
                        score_cutoff=0
                    )[1],
                    rapidfuzz_process.extractOne(
                        f"{hotword} {context.next_word.word}",
                        [hotword],
                        scorer=Levenshtein.normalized_similarity,
                        score_cutoff=0
                    )[1]
                )
                context_score += neighbor_match * 0.2
            
            final_score = base_score * context_score
            if final_score >= score_cutoff/100:
                matches.append((hotword, final_score))
    
    if matches:
        # Return the match with highest score
        return max(matches, key=lambda x: x[1])
        
    return None, 0.0 