"""Streaming synthesis for Chatterbox Turbo.

Since S3Gen's CFM decoder requires the full token sequence (encoder upsampling
creates length dependencies between prompt and generated tokens), we can't do
true token-level chunked audio streaming.

Instead, we do sentence-level streaming: split text into sentences, generate
each one fully (T3 tokens → S3Gen decode), and yield audio per sentence.
This gives meaningful TTFA improvement on multi-sentence inputs.

Watermarking is skipped for streaming (requires full waveform).
"""

import logging
import re
from typing import Generator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

_LOGGER = logging.getLogger(__name__)

# Regex to split text into sentences
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns at least one item."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences if sentences else [text]


def generate_streaming(
    model,
    text: str,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    **kwargs,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Sentence-level streaming synthesis for Chatterbox Turbo.

    Splits text into sentences and yields audio for each sentence as it completes.
    For single-sentence inputs, this behaves the same as non-streaming.

    Yields (audio_chunk_float32, sample_rate) tuples.
    """
    from chatterbox.tts_turbo import punc_norm
    from chatterbox.models.s3gen.const import S3GEN_SIL

    assert model.conds is not None, "Voice conditionals must be set before streaming"

    sr = model.sr
    device = model.device

    sentences = _split_sentences(text)
    _LOGGER.debug("Streaming %d sentence(s)", len(sentences))

    for sentence in sentences:
        sentence = punc_norm(sentence)
        text_tokens = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(device)

        # Generate all speech tokens for this sentence
        speech_tokens = model.t3.inference_turbo(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Clean up tokens
        speech_tokens = speech_tokens[speech_tokens < 6561].to(device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], device=device).long()
        speech_tokens = torch.cat([speech_tokens, silence])

        # Decode to audio (single pass, no watermark)
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
        )
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

        yield audio, sr
