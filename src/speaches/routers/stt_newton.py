import asyncio
from collections.abc import Generator, Iterable, AsyncGenerator
import logging
from typing import Annotated, Literal, Optional

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from faster_whisper.transcribe import BatchedInferencePipeline, TranscriptionInfo
from huggingface_hub.utils._cache_manager import _scan_cached_repo

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
    TranscriptionWord,
)
from speaches.dependencies import AudioFileDependency, ConfigDependency, WhisperModelManagerDependency
from speaches.executors.whisper import utils as whisper_utils
from speaches.hf_utils import get_model_card_data_from_cached_repo_info, get_model_repo_path
from speaches.model_aliases import ModelId
from speaches.text_utils import segments_to_srt, segments_to_text, segments_to_vtt
from speaches.word_processing import (
    get_word_context,
    should_process_word,
    find_best_hotword_match,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]

# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"


def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    match response_format:
        case "text":
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(),
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info).model_dump_json(),
                media_type="application/json",
            )
        case "vtt":
            return Response(
                "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
            )
        case "srt":
            return Response(
                "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
            )


def format_as_sse(data: str) -> str:
    return f"{data}\n\n"


async def process_segment(
    segment: TranscriptionSegment,
    hotwords: str | None = None,
    score_cutoff: float = 85.0,
    confidence_threshold: float = 0.8
) -> TranscriptionSegment:
    """Post-process a single transcription segment using fuzzy matching with context awareness.
    
    Args:
        segment: The segment to process
        hotwords: Optional comma-separated string of hotwords to apply
        score_cutoff: Minimum similarity score (0-100) for fuzzy matching
        confidence_threshold: Base confidence threshold for word processing
        
    Returns:
        Processed segment with modified words
    """
    if not hotwords or not segment.words:
        return segment
        
    hotword_list = [hw.strip() for hw in hotwords.split(",")]
    processed_words = []
    modified = False
    
    for i, word in enumerate(segment.words):
        context = get_word_context(segment.words, i)
        
        if should_process_word(word, context, confidence_threshold):
            best_match, score = find_best_hotword_match(
                word,
                context,
                hotword_list,
                score_cutoff
            )
            
            if best_match:
                # Create new word with the same timing but updated text and probability
                new_word = TranscriptionWord(
                    start=word.start,
                    end=word.end,
                    word=best_match,
                    # Combine original confidence with match score and context
                    probability=word.probability * score * (
                        1.0 + (context.avg_segment_confidence - word.probability) * 0.2
                    )
                )
                processed_words.append(new_word)
                modified = True
                continue
        
        processed_words.append(word)
    
    if modified:
        # Create new segment with modified words
        new_segment = segment.model_copy()
        new_segment.words = processed_words
        # Update segment text to reflect word changes
        new_segment.text = " ".join(word.word for word in processed_words)
        return new_segment
    
    return segment


def segments_to_streaming_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
    hotwords: str | None = None,
    score_cutoff: float = 85.0,
    confidence_threshold: float = 0.8,
) -> StreamingResponse:
    async def segment_responses() -> AsyncGenerator[str, None]:
        for i, segment in enumerate(segments):
            # Apply post-processing to each segment as it arrives
            processed_segment = await process_segment(
                segment, 
                hotwords, 
                score_cutoff,
                confidence_threshold
            )
            
            if response_format == "text":
                data = processed_segment.text
            elif response_format == "json":
                data = CreateTranscriptionResponseJson.from_segments([processed_segment]).model_dump_json()
            elif response_format == "verbose_json":
                data = CreateTranscriptionResponseVerboseJson.from_segment(
                    processed_segment, transcription_info
                ).model_dump_json()
            elif response_format == "vtt":
                data = segments_to_vtt(processed_segment, i)
            elif response_format == "srt":
                data = segments_to_srt(processed_segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


@router.post(
    "/v1/audio/newtonTranslations",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def translate_file(
    config: ConfigDependency,
    model_manager: WhisperModelManagerDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    vad_filter: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=vad_filter,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)

        if stream:
            return segments_to_streaming_response(segments, transcription_info, response_format)
        else:
            return segments_to_response(segments, transcription_info, response_format)


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities


 # https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/newtonTranscriptions",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def transcribe_file(
    config: ConfigDependency,
    model_manager: WhisperModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    score_cutoff: Annotated[float, Form()] = 85.0,
    confidence_threshold: Annotated[float, Form()] = 0.8,
    vad_filter: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    model_repo_path = get_model_repo_path(model)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    assert model_card_data is not None, cached_repo_info  # FIXME
    if whisper_utils.hf_model_filter.passes_filter(model_card_data):
        with model_manager.load_model(model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
            segments, transcription_info = whisper_model.transcribe(
                audio,
                task="transcribe",
                language=language,
                initial_prompt=prompt,
                word_timestamps="word" in timestamp_granularities,
                vad_filter=vad_filter,
                hotwords=hotwords,
            )
            # Convert faster-whisper segments to our format
            segments = TranscriptionSegment.from_faster_whisper_segments(segments)

            if stream:
                return segments_to_streaming_response(
                    segments, 
                    transcription_info, 
                    response_format, 
                    hotwords,
                    score_cutoff,
                    confidence_threshold
                )
            else:
                processed_segments = []
                for segment in segments:
                    processed_segment = asyncio.run(process_segment(
                        segment, 
                        hotwords,
                        score_cutoff,
                        confidence_threshold
                    ))
                    processed_segments.append(processed_segment)
                return segments_to_response(processed_segments, transcription_info, response_format)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not supported. If you think this is a mistake, please open an issue.",
        )
